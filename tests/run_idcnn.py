import argparse
import os
from nlp_architect.procedures import TrainTagger

train_count = 14987
# params = {"labeled_precentage": [0.01, 0.02, 0.05, 0.1]}
params = {"labeled_precentage": [1]}
unlabeled_count = 10000
labeled_count = train_count - unlabeled_count
data_dir = "/home/shira_nlp/bert/data/CoNLL"
splits_dir = data_dir + os.path.sep + "labeled_splits"
max_sentence_length = 50
max_word_length = 12
b = 100

precentage_str = '1'

idcnn_crf_output_path = '/home/shira_nlp/idcnn_models' + os.sep + precentage_str
# log_path = '/home/shira_nlp/logs/train_idcnn.txt'
if not os.path.exists(idcnn_crf_output_path):
    os.makedirs(idcnn_crf_output_path)

parser = argparse.ArgumentParser()
train_procedure = TrainTagger()
train_procedure.add_arguments(parser)
docstart_token = "-DOCSTART-"
# embeddings_path =  '/home/peter_nlp/data/glove.6B/glove.6B.100d.txt'
embeddings_path = '/home/shira_nlp/dilated-cnn-ner/data/embeddings/lample-embeddings-pre_orig.txt'

# run idcnn
for i in range(2):
    log = '/home/shira_nlp/logs/idcnn_{}.log'.format(str(i))
    args = parser.parse_args(
            [
                '--data_dir', data_dir, '--output_dir', idcnn_crf_output_path,
                '--embedding_file', embeddings_path,
                '--config_file', '/home/shira_nlp/idcnn_experiments/config_files/replicate_config.json', '-b', '8',
                '--lr', '0.0005', '-e', '5000', '--model_type', 'id-cnn', '--best_result_file', log, '--max_sentence_length',
                '50', '--max_word_length', '12', '--logging_steps', '300', '--save_steps', '0',
                '--word_dropout', '0', '--ignore_token=-DOCSTART-', '--train_filename', '0_0_1_labeled.txt',
                '--overwrite_output_dir'
            ])

    # # run lstm
    # args = parser.parse_args(
    #         [
    #             '--data_dir', data_dir, '--output_dir', idcnn_crf_output_path,
    #             '--embedding_file', embeddings_path,
    #             '--config_file', '/home/shira_nlp/idcnn_experiments/config_files/lstm_config.json', '-b', '128',
    #             '--lr', '0.0008', '-e', '300', '--model_type', 'cnn-lstm', '--log_file', log_path, '--max_sentence_length',
    #             '50', '--max_word_length', '12', '--logging_steps', '500', '--ignore_token=-DOCSTART-',
    #             '--bilou', '--overwrite_output_dir'
    #         ])

    train_procedure.run_procedure(args)    