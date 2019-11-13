import argparse
import os
from nlp_architect.procedures.transformers.que_ans import (
    TransformerQuestionAnsweringTrain, TransformerQuestionAnsweringRun)


def test_qa():
    data_dir = 'SQUAD'
    bert_output_path = data_dir + os.sep + 'bert_out'
    # train bert qa
    parser = argparse.ArgumentParser()
    train_proc = TransformerQuestionAnsweringTrain()
    train_proc.add_arguments(parser)
    if not os.path.exists(bert_output_path):
        os.makedirs(bert_output_path)
    args = parser.parse_args(
        [
            '--data_dir', data_dir, '--output_dir', bert_output_path,
            '--model_name_or_path', 'bert-large-uncased-whole-word-masking',
            '--model_type', 'bert', '--num_train_epochs', '1', '--per_gpu_train_batch_size',
            '8', '--overwrite_output_dir', '--do_lower_case'
        ])
    #'--version_2_with_negative'
    train_proc.run_procedure(args)

    # inference
    data_file = data_dir + os.sep + 'test_inf.json'
    model_path = bert_output_path
    output_dir = bert_output_path + os.sep + 'inference'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    parser = argparse.ArgumentParser()
    inf_proc = TransformerQuestionAnsweringRun()
    inf_proc.add_arguments(parser)
    args = parser.parse_args(
        [
            '--data_file', data_file, '--model_path', model_path,
            '--model_type', 'bert', '--output_dir', output_dir
        ])
    inf_proc.run_procedure(args)

if __name__ == "__main__":
    test_qa()
