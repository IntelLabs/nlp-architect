import argparse
import math
import os
import shutil
from subprocess import run
from nlp_architect.procedures.transformers.que_ans import TransformerQuestionAnsweringTrain


def test_qa():
    
    data_dir = 'SQUAD'
    bert_output_path = data_dir + os.sep + 'bert_out'
    # test bert qa
    parser = argparse.ArgumentParser()
    train_proc = TransformerQuestionAnsweringTrain()
    train_proc.add_arguments(parser)
    if not os.path.exists(bert_output_path):
        os.makedirs(bert_output_path)
    args = parser.parse_args(
        ['--data_dir', data_dir, '--output_dir',
            bert_output_path, '--model_name_or_path',
            'bert-large-uncased-whole-word-masking', '--model_type', 'bert',
            '--num_train_epochs', '2', '--per_gpu_train_batch_size', '3', '--eval_script',
            'squad_eval.py', '--overwrite_output_dir','--do_lower_case'])
            #'--version_2_with_negative'
    train_proc.run_procedure(args)

if __name__== "__main__":
    test_qa()