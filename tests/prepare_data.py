
from sklearn.model_selection import ParameterGrid
from nlp_architect.procedures.transformers.seq_tag import TransformerTokenClsTrain
import argparse
import math
import os
from subprocess import run
from nlp_architect.data.sequential_tagging import TokenClsProcessor
from nlp_architect.models.tagging import NeuralTagger
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from nlp_architect.nn.torch import setup_backend
from nlp_architect.data.utils import split_column_dataset
from tests.utils import count_examples

train_count = 14987
params = {"labeled_precentage": [0.2]}
unlabeled_count = 10000
labeled_count = train_count - unlabeled_count
data_dir = "/home/shira_nlp/bert/data/CoNLL/"
splits_dir = data_dir + os.path.sep + "data_splits_bilou"
max_sentence_length = 50
max_word_length = 12



# classifier = NeuralTagger.load_model(model_path='/home/shira_nlp/cnn_lstm_output/')
# device, _ = setup_backend(False)
# classifier.to(device)



# processor = TokenClsProcessor(data_dir)
# dev_ex = processor.get_dev_examples()
# test_ex = processor.get_test_examples()

# dev_dataset = classifier.convert_to_tensors(dev_ex,
#                                             max_seq_length=max_sentence_length,
#                                             max_word_length=max_word_length)
# dev_sampler = SequentialSampler(dev_dataset)
# dev_dl = DataLoader(dev_dataset, sampler=dev_sampler,
#                     batch_size=b)

# test_dataset = classifier.convert_to_tensors(test_ex,
#                                               max_seq_length=max_sentence_length,
#                                               max_word_length=max_word_length)
# test_sampler = SequentialSampler(test_dataset)
# test_dl = DataLoader(test_dataset, sampler=test_sampler,
#                       batch_size=b)

# classifier._get_eval(dev_dl, "dev")
# classifier._get_eval(test_dl, "test")


# prepare data for experiments

# prepare fixed unlabeled data
# split_column_dataset(first_count=unlabeled_count, second_count=labeled_count, dataset=data_dir+'/train.txt',first_filename='10000_unlabeled_bilou.txt',
#                                   second_filename='rest_train_bilou.txt', out_folder=data_dir)
# assert count_examples(data_dir+'/10000_unlabeled_bilou.txt') == 10000
# assert count_examples(data_dir+'/rest_train_bilou.txt') == labeled_count

runs_count = 20

for p in ParameterGrid(params):
    precentage_str = str(p['labeled_precentage'])
    print(precentage_str)
    split_directory = splits_dir + os.sep + precentage_str
    if not os.path.exists(split_directory):
        os.makedirs(split_directory)
    for r in range(runs_count):
        run_directory = split_directory + os.sep + str(r+1)
        if not os.path.exists(run_directory):
            os.makedirs(run_directory)
        # split dataset
        # processor = TokenClsProcessor(data_dir)
        split_column_dataset(first_count=math.ceil(train_count * p['labeled_precentage']), second_count=0, \
        dataset=data_dir+'/rest_train_bilou.txt',first_filename='labeled.txt',
                                second_filename='', out_folder=run_directory)
        assert count_examples(run_directory+'/labeled.txt') == math.ceil(train_count * p['labeled_precentage'])
                        
    
# # for each experiment, train bert into a folder, then run distillation and write best results into a file
# # 
# runs_count = 20
# for p in ParameterGrid(params):
#   precentage_str = str(p['labeled_precentage'])
#   split_directory = splits_dir + os.sep + precentage_str
#   if os.path.exists(split_directory):
#     bert_output_path = '/home/shira_nlp/bert_ner_models' + os.sep + precentage_str
#     distil_softmax_output_path = '/home/shira_nlp/distil_ner_models' + os.sep + precentage_str
#     distil_crf_output_path = '/home/shira_nlp/distil_ner_crf_models' + os.sep + precentage_str
#     os.makedirs(bert_output_path)
#     os.makedirs(distil_softmax_output_path)
#     os.makedirs(distil_crf_output_path)
#     for exp in range(runs_count):
#       run_directory = split_directory + os.sep + str(exp+1)
#       labeled_data = os.sep + run_directory + os.sep + 'labeled.txt'
#       if os.path.exists(labeled_data):
#         # prepare output folders for bert/distilled model
#         bert_output_path = bert_output_path + os.sep + str(exp+1)
#         distil_softmax_output_path = distil_softmax_output_path + os.sep + str(exp+1)
#         distil_crf_output_path = distil_crf_output_path + os.sep + str(exp+1)
#         os.makedirs(bert_output_path)
#         os.makedirs(distil_softmax_output_path)
#         os.makedirs(distil_crf_output_path)
#         train_file_name = '/labeled_splits' + os.sep + precentage_str+ os.sep + str(exp+1) + '/labeled.txt'
#         unlabeled_file_name = 'unlabeled.txt'
#         # train bert
#         cmd_str = 'nlp_architect train transformer_token --data_dir={} '\
#               '--model_name_or_path={} --model_type={} --output_dir={} '\
#               '--train_file_name={} --num_train_epoch={} --overwrite_output_dir'\
#                 .format(data_dir, 'bert-base-cased', 'bert', bert_output_path, train_file_name, '5')
#         print(cmd_str)
#         run(cmd_str, shell=True)
#         # train distilled ner and write output
#         # train CNN-LSTM-softmax
#         cmd_str = 'nlp_architect train tagger_kd_pseudo --data_dir={} --output_dir={} '\
#               '--teacher_model_path={} --teacher_model_type={} --embedding_file={} '\
#                 '--labeled_train_file={} --unlabeled_train_file={} -b={} -b_ul={} -e={} '\
#                   '--lr={} --overwrite_output_dir'\
#                     .format(data_dir, distil_softmax_output_path, bert_output_path, 'bert',
#                     '/home/peter_nlp/data/glove.6B/glove.6B.100d.txt', train_file_name,
#                     '32','30','20','0.001')
#         # train CNN-LSTM-CRF
#         cmd_str = 'nlp_architect train tagger_kd_pseudo --data_dir={} --output_dir={} '\
#               '--teacher_model_path={} --teacher_model_type={} --embedding_file={} '\
#                 '--labeled_train_file={} --unlabeled_train_file={} -b={} -b_ul={} -e={} '\
#                   '--lr={} --use_crf --overwrite_output_dir'\
#                     .format(data_dir, distil_softmax_output_path, bert_output_path, 'bert',
#                     '/home/peter_nlp/data/glove.6B/glove.6B.100d.txt', train_file_name,
#                     '32','30','20','0.001')



    # cmd_str = 'cp /home/shira_nlp/bert/data/CoNLL/labeled.txt ' + new_directory + os.sep + 
    # print(cmd_str)
    # run(cmd_str, shell=True)
        
    

    # # cmd_str = 'nlp_architect train tagger_kd_pseudo --data_dir={} '\
    # #       '--output_dir={} --teacher_model_path={} --teacher_model_type={} ----overwrite_output_dir'\
    # #         .format('/home/shira_nlp/split_output/',\
    # #         '/home/shira_nlp/distillation_output', '/home/shira_nlp/output/', 'bert')


