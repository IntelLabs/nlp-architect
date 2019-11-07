import argparse
import math
import os
import shutil
from nlp_architect.procedures.transformers.seq_tag import TransformerTokenClsTrain
from nlp_architect.procedures.token_tagging import TrainTaggerKDPseudo
from nlp_architect.data.sequential_tagging import TokenClsProcessor


def test_distillation():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(current_dir, 'fixtures/data/distillation')
    num_of_examples = count_examples(data_dir + os.sep + 'train.txt')
    labeled_precentage = 0.4
    unlabeled_precentage = 0.5
    bert_output_path = data_dir + os.sep + 'bert_out'
    kd_output_path = data_dir + os.sep + 'kd_out'
    assert(labeled_precentage <= 1 and unlabeled_precentage <= 1)

    # test split dataset
    if os.path.exists(data_dir):
        processor = TokenClsProcessor(data_dir)
        processor.split_dataset(
            dataset='train.txt', labeled=math.ceil(num_of_examples * labeled_precentage),
            unlabeled=math.ceil(num_of_examples * unlabeled_precentage), out_folder=data_dir)
        check_labeled_count = count_examples(data_dir + os.sep + 'labeled.txt')
        assert(check_labeled_count == math.ceil(num_of_examples * labeled_precentage))
        check_unlabeled_count = count_examples(data_dir + os.sep + 'unlabeled.txt')
        assert(check_unlabeled_count == math.ceil(num_of_examples * unlabeled_precentage))

    # test bert ner
    parser = argparse.ArgumentParser()
    train_proc = TransformerTokenClsTrain()
    train_proc.add_arguments(parser)
    if not os.path.exists(bert_output_path):
        os.makedirs(bert_output_path)
    args = parser.parse_args(
        ['--data_dir', data_dir, '--output_dir',
            bert_output_path, '--model_name_or_path',
            'bert-base-cased', '--model_type', 'bert',
            '--train_file_name', 'labeled.txt', '--num_train_epoch',
            '1', '--per_gpu_train_batch_size', '50', '--overwrite_output_dir'])
    train_proc.run_procedure(args)
    assert(os.path.exists(bert_output_path + os.sep + 'pytorch_model.bin'))

    # test distillation pseudo
    parser = argparse.ArgumentParser()
    train_proc = TrainTaggerKDPseudo()
    train_proc.add_arguments(parser)
    out_dir = kd_output_path
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    args = parser.parse_args(
        ['--data_dir', data_dir, '--output_dir', out_dir,
            '--teacher_model_path', bert_output_path,
            '--teacher_model_type', 'bert', '--labeled_train_file',
            'labeled.txt', '--unlabeled_train_file', 'unlabeled.txt',
            '-e', '10', '--overwrite_output_dir'])
    train_proc.run_procedure(args)
    assert(os.path.exists(out_dir + os.sep + 'model.bin'))

    # delete test files
    if os.path.exists(bert_output_path):
        shutil.rmtree(bert_output_path)
    if os.path.exists(kd_output_path):
        shutil.rmtree(kd_output_path)
    if os.path.exists(data_dir + os.sep + 'labeled.txt'):
        os.remove(data_dir + os.sep + 'labeled.txt')
    if os.path.exists(data_dir + os.sep + 'unlabeled.txt'):
        os.remove(data_dir + os.sep + 'unlabeled.txt')


def count_examples(file):
    ctr = 0
    if os.path.exists(file):
        with open(file) as fp:
            for line in fp:
                line = line.strip()
                if len(line) == 0:
                    ctr += 1
    else:
        print("File:" + file + " doesn't exist")
    return ctr
