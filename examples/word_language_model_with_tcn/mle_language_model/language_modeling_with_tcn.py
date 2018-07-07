"""
This script replicates the experiments run in the following paper for music data.
Bai, Shaojie, J. Zico Kolter, and Vladlen Koltun. "An Empirical Evaluation of Generic Convolutional
and Recurrent Networks for Sequence Modeling." arXiv preprint arXiv:1803.01271 (2018).

To compare with the original implementation, run
python ./language_modeling_with_tcn.py --batch_size 16 --dropout 0.45 --epochs 100 --ksize 3
--levels 4 --seq_len 60 --nhid 600 --em_len 600 --em_dropout 0.25 --lr 4 --grad_clip_value 0.35
--results_dir ./ --dataset PTB

python ./language_modeling_with_tcn.py --batch_size 16 --dropout 0.5 --epochs 100 --ksize 3
--levels 5 --seq_len 60 --nhid 1000 --em_len 400 --em_dropout 0.25 --lr 4 --grad_clip_value 0.35
--results_dir ./ --dataset WikiText-103

"""
import os
import argparse
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from nlp_architect.data.ptb import PTBDataLoader, PTBDictionary
from nlp_architect.models.temporal_convolutional_network import TCN, CommonLayers
from nlp_architect.utils.io import validate, validate_existing_directory, \
    validate_existing_filepath, validate_parent_exists, check_size


class TCNForLM(TCN, CommonLayers):
    """
    Main class that defines training graph and defines training run method for language modeling
    """
    def __init__(self, *args, **kwargs):
        super(TCNForLM, self).__init__(*args, **kwargs)
        self.num_words = None
        self.input_placeholder_tokens = None
        self.label_placeholder_tokens = None
        self.learning_rate = None
        self.input_embeddings = None
        self.prediction = None
        self.projection_out = None
        self.gen_seq_prob = None
        self.training_loss = None
        self.validation_loss = None
        self.test_loss = None
        self.merged_summary_op_train = None
        self.merged_summary_op_test = None
        self.merged_summary_op_val = None
        self.training_update_step = None

    def run(self, data_loaders, lr, num_iterations=100, log_interval=100, result_dir="./",
            ckpt=None):
        """
        Args:
            data_loaders: dict, keys are "train", "valid", "test", values are corresponding iterator
                                dataloaders
            lr: float, learning rate
            num_iterations: int, number of iterations to run
            log_interval: int, number of iterations after which to run validation and log
            result_dir: str, path to results directory
            ckpt: str, location of checkpoint file

        Returns:
            None
        """

        summary_writer = tf.summary.FileWriter(os.path.join(result_dir, "tfboard"),
                                               tf.get_default_graph())
        saver = tf.train.Saver(max_to_keep=None)

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

        if ckpt is not None:
            saver.restore(sess, ckpt)

        all_vloss = []
        for i in range(num_iterations):

            x_data, y_data = next(data_loaders["train"])

            feed_dict = {self.input_placeholder_tokens: x_data,
                         self.label_placeholder_tokens: y_data, self.training_mode: True,
                         self.learning_rate: lr}
            _, summary_train, total_loss_i = sess.run([self.training_update_step,
                                                       self.merged_summary_op_train,
                                                       self.training_loss],
                                                      feed_dict=feed_dict)

            summary_writer.add_summary(summary_train, i)

            if i % log_interval == 0:
                print("Step {}: Total: {}".format(i, total_loss_i))
                saver.save(sess, result_dir, global_step=i)

                val_loss = {}
                for split_type in ["valid", "test"]:
                    val_loss[split_type] = 0
                    data_loaders[split_type].reset()
                    count = 0
                    for x_data_test, y_data_test in data_loaders[split_type]:
                        feed_dict = {self.input_placeholder_tokens: x_data_test,
                                     self.label_placeholder_tokens: y_data_test,
                                     self.training_mode: False}
                        val_loss[split_type] += sess.run(self.training_loss, feed_dict=feed_dict)
                        count += 1

                    val_loss[split_type] = val_loss[split_type] / count

                summary_val = sess.run(self.merged_summary_op_val,
                                       feed_dict={self.validation_loss: val_loss["valid"]})
                summary_test = sess.run(self.merged_summary_op_test,
                                        feed_dict={self.test_loss: val_loss["test"]})

                summary_writer.add_summary(summary_val, i)
                summary_writer.add_summary(summary_test, i)

                print("Validation loss: {}".format(val_loss["valid"]))
                print("Test loss: {}".format(val_loss["test"]))
                all_vloss.append(val_loss["valid"])

                if i > 3 * log_interval and val_loss["valid"] >= max(all_vloss[-5:]):
                    lr = lr / 2.

    def run_inference(self, ckpt, num_samples=10, sos=0, eos=1):
        """
        Method for running inference for generating sequences
        Args:
            ckpt: Location of checkpoint file with trained model
            num_samples: int, number of samples to generate
            sos: int, start of sequence symbol
            eos: int, end of sequence symbol

        Returns:
            List of sequences
        """
        saver = tf.train.Saver(max_to_keep=None)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        if ckpt is not None:
            saver.restore(sess, ckpt)

        results = self.sample_sequence(sess, num_samples, sos=sos, eos=eos)
        return results

    def build_train_graph(self, num_words=20000, word_embeddings=None, max_gradient_norm=None,
                          em_dropout=0.4):
        """
        Method that builds the graph for training
        Args:
            num_words: int, number of words in the vocabulary
            word_embeddings: numpy array, optional numpy array to initialize embeddings
            max_gradient_norm: float, maximum gradient norm value for clipping
            em_dropout: float, dropout rate for embeddings

        Returns:
            None
        """
        self.num_words = num_words
        with tf.variable_scope("input", reuse=True):
            self.input_placeholder_tokens = tf.placeholder(tf.int32, [None, self.max_len],
                                                           name='input_tokens')
            self.label_placeholder_tokens = tf.placeholder(tf.int32, [None, self.max_len],
                                                           name='input_tokens_shifted')
            self.learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')

        self.input_embeddings = self.define_input_layer(self.input_placeholder_tokens,
                                                        word_embeddings,
                                                        embeddings_trainable=True)

        with tf.device('/gpu:0'):
            input_embeddings_dropped = tf.layers.dropout(self.input_embeddings,
                                                         rate=em_dropout,
                                                         training=self.training_mode)
            self.prediction = self.build_network_graph(input_embeddings_dropped,
                                                       last_timepoint=False)

        if self.prediction.shape[-1] != self.n_features_in:
            print("Not tying weights")
            tied_weights = False
        else:
            print("Tying weights")
            tied_weights = True
        self.projection_out = self.define_projection_layer(self.prediction,
                                                           tied_weights=tied_weights)
        self.gen_seq_prob = tf.nn.softmax(self.projection_out)

        with tf.device('/gpu:0'):
            with tf.variable_scope("training"):
                params = tf.trainable_variables()

                soft_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.label_placeholder_tokens, logits=self.projection_out)
                ce_last_tokens = tf.slice(soft_ce, [0, int(self.max_len / 2)],
                                          [-1, int(self.max_len / 2)])
                self.training_loss = tf.reduce_mean(ce_last_tokens)

                summary_ops_train = [tf.summary.scalar("Training Loss", self.training_loss),
                                     tf.summary.scalar("Training perplexity",
                                                       tf.exp(self.training_loss))]
                self.merged_summary_op_train = tf.summary.merge(summary_ops_train)

                self.validation_loss = tf.placeholder(tf.float32, shape=())
                summary_ops_val = [tf.summary.scalar("Validation Loss", self.validation_loss),
                                   tf.summary.scalar("Validation perplexity",
                                                     tf.exp(self.validation_loss))]
                self.merged_summary_op_val = tf.summary.merge(summary_ops_val)

                self.test_loss = tf.placeholder(tf.float32, shape=())
                summary_ops_test = [tf.summary.scalar("Test Loss", self.test_loss),
                                    tf.summary.scalar("Test perplexity", tf.exp(self.test_loss))]
                self.merged_summary_op_test = tf.summary.merge(summary_ops_test)

                # Calculate and clip gradients
                gradients = tf.gradients(self.training_loss, params)

                if max_gradient_norm is not None:
                    clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
                else:
                    clipped_gradients = gradients

                grad_norm = tf.global_norm(clipped_gradients)
                summary_ops_train.append(tf.summary.scalar("Grad Norm", grad_norm))

                # Optimization
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                summary_ops_train.append(tf.summary.scalar("Learning rate", self.learning_rate))
                self.merged_summary_op_train = tf.summary.merge(summary_ops_train)

                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
                with tf.control_dependencies(update_ops):
                    self.training_update_step = optimizer.apply_gradients(zip(clipped_gradients,
                                                                              params))

    def sample_sequence(self, sess, num_samples=10, sos=0, eos=1):
        """
        Method for sampling a sequence (repeatedly one symbol at a time)
        Args:
            sess: tensorflow session
            num_samples: int, number of samples to generate
            sos: int, start of sequence symbol
            eos: int, end of sequence symbol

        Returns:
            List of sequences
        """
        all_sequences = []
        for _ in tqdm(range(num_samples)):
            sampled_sequence = []
            input_sequence = sos * np.ones((1, self.max_len))
            count = 0
            elem = sos
            while (elem != eos) and (count <= self.max_len * 10):
                feed_dict = {self.input_placeholder_tokens: input_sequence,
                             self.training_mode: False}
                gen_seq_prob_value = sess.run(self.gen_seq_prob, feed_dict=feed_dict)
                prob = gen_seq_prob_value[0, -1, :].astype(np.float64)
                prob = prob / sum(prob)
                elem = np.where(np.random.multinomial(1, prob))[0][0]
                input_sequence = np.roll(input_sequence, -1, axis=-1)
                input_sequence[:, -1] = elem
                count += 1
                sampled_sequence.append(elem)
            all_sequences.append(sampled_sequence)
        return all_sequences


def main(args):
    """
    Main function
    Args:
        args: output of argparse with all input arguments

    Returns:
        None
    """
    hidden_sizes = [args.nhid] * args.levels
    kernel_size = args.ksize
    dropout = args.dropout
    em_dropout = args.em_dropout
    seq_len = args.seq_len
    batch_size = args.batch_size
    n_epochs = args.epochs
    embedding_size = args.em_len
    datadir = os.path.abspath(args.datadir)
    results_dir = os.path.abspath(args.results_dir)

    ptb_dict = PTBDictionary(data_dir=datadir, dataset=args.dataset)
    ptb_dataset_train = PTBDataLoader(ptb_dict, data_dir=datadir, seq_len=seq_len,
                                      split_type="train", skip=seq_len / 2, dataset=args.dataset)
    ptb_dataset_valid = PTBDataLoader(ptb_dict, data_dir=datadir, seq_len=seq_len,
                                      split_type="valid", loop=False, skip=seq_len / 2,
                                      dataset=args.dataset)
    ptb_dataset_test = PTBDataLoader(ptb_dict, data_dir=datadir, seq_len=seq_len,
                                     split_type="test", loop=False, skip=seq_len / 2,
                                     dataset=args.dataset)

    n_train = ptb_dataset_train.n_train
    num_words = len(ptb_dataset_train.idx2word)
    n_per_epoch = int(n_train * 1.0 / batch_size)
    if not args.inference:
        num_iterations = int(n_train * n_epochs * 1.0 / batch_size)
        print("Training examples per epoch: {}, num iterations per epoch: {}"
              .format(n_train, num_iterations // n_epochs))
    else:
        num_iterations = 0

    with tf.device('/gpu:0'):
        model = TCNForLM(seq_len, embedding_size, hidden_sizes, kernel_size=kernel_size,
                         dropout=dropout)

        model.build_train_graph(num_words=num_words, max_gradient_norm=args.grad_clip_value,
                                em_dropout=em_dropout)

        if not args.inference:
            model.run(
                {"train": ptb_dataset_train, "valid": ptb_dataset_valid, "test": ptb_dataset_test},
                args.lr, num_iterations=num_iterations, log_interval=n_per_epoch,
                result_dir=results_dir, ckpt=None)
        else:
            sequences = model.run_inference(args.ckpt, num_samples=args.num_samples,
                                            sos=ptb_dict.sos_symbol, eos=ptb_dict.eos_symbol)
            for seq in sequences:
                sentence = []
                for idx in seq:
                    while idx == ptb_dict.sos_symbol:
                        continue
                    sentence.append(ptb_dict.idx2word[idx])
                print(" ".join(sentence) + "\n")


PARSER = argparse.ArgumentParser()
PARSER.add_argument('--seq_len', type=int,
                    help="Number of time points in each input sequence",
                    default=60)
PARSER.add_argument('--results_dir', type=validate_parent_exists,
                    help="Directory to write results to", default='./')
PARSER.add_argument('--dropout', type=float, default=0.45, action=check_size(0, 1),
                    help='dropout applied to layers, value in [0, 1] (default: 0.45)')
PARSER.add_argument('--ksize', type=int, default=3,
                    help='kernel size (default: 3)')
PARSER.add_argument('--levels', type=int, default=4,
                    help='# of levels (default: 4)')
PARSER.add_argument('--lr', type=float, default=4,
                    help='initial learning rate (default: 4)')
PARSER.add_argument('--nhid', type=int, default=600,
                    help='number of hidden units per layer (default: 600)')
PARSER.add_argument('--em_len', type=int, default=600,
                    help='Length of embedding (default: 600)')
PARSER.add_argument('--em_dropout', type=float, default=0.25, action=check_size(0, 1),
                    help='dropout applied to layers, value in [0, 1] (default: 0.25)')
PARSER.add_argument('--grad_clip_value', type=float, default=0.35,
                    help='value to clip each element of gradient')
PARSER.add_argument('--batch_size', type=int, default=16,
                    help='Batch size')
PARSER.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs')
PARSER.add_argument('--datadir', type=validate_existing_directory, default="../data/",
                    help='dir to download data if not already present')
PARSER.add_argument('--dataset', type=str, default="PTB", choices=['PTB', 'WikiText-103'],
                    help='dataset name')
PARSER.add_argument('--inference', action='store_true',
                    help='whether to run in inference mode')
PARSER.add_argument('--ckpt', type=validate_existing_filepath,
                    help='checkpoint file')
PARSER.add_argument('--num_samples', type=int, default=10,
                    help='number of samples to generate during inference')
PARSER.set_defaults()
ARGS = PARSER.parse_args()
main(ARGS)
