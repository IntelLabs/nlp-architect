# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
from __future__ import print_function, division

import argparse
import copy

import evaluate
import tensorflow as tf

from nlp_architect.data.fasttext_emb import FastTextEmb
from nlp_architect.models.crossling_emb import WordTranslator
from nlp_architect.utils.io import validate_existing_directory, validate_parent_exists, check_size

if __name__ == "__main__":

    print("\t\t" + 40 * "=")
    print("\t\t= Unsupervised Crosslingual Embeddings =")
    print("\t\t" + 40 * "=")

    # Parsing arguments for model parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dim", type=int, default=300,
                        help="Embedding Dimensions", action=check_size(1, 1024))
    parser.add_argument("--vocab_size", type=int, default=200000,
                        help="Vocabulary Size", action=check_size(1, 1000000))
    parser.add_argument("--lr", type=float, default=0.1, help="Learning Rate",
                        action=check_size(0.00001, 2.0))
    parser.add_argument("--beta", type=float, default=0.001, help="Beta for W orthogornaliztion",
                        action=check_size(0.0000001, 5.0))
    parser.add_argument("--smooth_val", type=float, default=0.1, help="Label smoother for\
                        discriminator", action=check_size(0.0001, 0.2))
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size", action=check_size(8, 1024))
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs to run", action=check_size(1, 20))
    parser.add_argument("--iters_epoch", type=int, default=1000000, help="Iterations to run\
                        each epoch", action=check_size(1, 2000000))
    parser.add_argument("--disc_runs", type=int, default=5, help="Number of times\
                        discriminator is run each iteration", action=check_size(1, 20))
    parser.add_argument("--most_freq", type=int, default=75000, help="Number of words to\
                        show discriminator", action=check_size(1, 1000000))
    parser.add_argument("--src_lang", type=str, default="en", help="Source Language",
                        action=check_size(1, 3))
    parser.add_argument("--tgt_lang", type=str, default="fr", help="Target Language",
                        action=check_size(1, 3))
    parser.add_argument("--data_dir", default=None, help="Data path for training and\
                        and evaluation data", type=validate_existing_directory)
    parser.add_argument("--eval_dir", default=None, help="Path for eval words",
                        type=validate_existing_directory)
    parser.add_argument("--weight_dir", default=None, help="path to save mapping\
                        weights", type=validate_parent_exists)
    hparams = parser.parse_args()
    # Load Source Embeddings
    src = FastTextEmb(hparams.data_dir, hparams.src_lang, hparams.vocab_size)
    src_dict, src_vec = src.load_embeddings()
    # Load Target Embeddings
    tgt = FastTextEmb(hparams.data_dir, hparams.tgt_lang, hparams.vocab_size)
    tgt_dict, tgt_vec = tgt.load_embeddings()

    # GAN instance
    train_model = WordTranslator(hparams, src_vec, tgt_vec, hparams.vocab_size)

    # Copy embeddings
    src_vec_eval = copy.deepcopy(src_vec)
    tgt_vec_eval = copy.deepcopy(tgt_vec)

    # Evaluator instance
    eval_model = evaluate.Evaluate(train_model.generator.W, src_vec_eval, tgt_vec_eval,
                                   src_dict, tgt_dict, hparams.src_lang, hparams.tgt_lang,
                                   hparams.eval_dir, hparams.vocab_size)

    # Tensorflow session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        local_lr = hparams.lr
        for epoch in range(hparams.epochs):
            # Train the model
            train_model.run(sess, local_lr)
            # Evaluate using nearest neighbors measure
            eval_model.calc_nn_acc(sess)
            # Evaluate using CSLS similarity measure
            eval_model.run_csls_metrics(sess)
            # Drop learning rate
            local_lr = train_model.set_lr(local_lr, eval_model.drop_lr)
            # Save model if it is good
            train_model.save_model(eval_model.save_model, sess)
            print("End of epoch " + str(epoch))
        # Apply procrustes to improve CSLS score
        final_pairs = eval_model.generate_dictionary(sess, dict_type="S2T&T2S")
        train_model.apply_procrustes(sess, final_pairs)
        # Run metrics to see improvement
        eval_model.run_csls_metrics(sess)
        # Save the model if there is imporvement
        train_model.save_model(eval_model.save_model, sess)
        # Write cross lingual embeddings to file
        train_model.generate_xling_embed(sess, src_dict, tgt_dict, tgt_vec)
    print("Completed Training")
