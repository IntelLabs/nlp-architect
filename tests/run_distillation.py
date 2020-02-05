
import argparse
import os
import math
from shutil import copyfile
from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader, SequentialSampler
from nlp_architect.procedures.transformers.seq_tag import TransformerTokenClsTrain
from nlp_architect.procedures.token_tagging import TrainTaggerKD, TrainTaggerKDPseudo
from nlp_architect.procedures import TrainTagger
from nlp_architect.models.transformers import TransformerTokenClassifier
from nlp_architect.data.sequential_tagging import TokenClsProcessor
from nlp_architect.nn.torch import setup_backend, set_seed
from tests.utils import count_examples


def train_tagger(tagger_type, data_dir, output_dir, embedding_file, config_file, batch_size, lr, epochs, best_dev_file,
                    logging_steps, ignore_token, train_filename):

    tagger_parser = argparse.ArgumentParser()
    tagger_procedure = TrainTagger()
    tagger_procedure.add_arguments(tagger_parser)
    tagger_args = tagger_parser.parse_args(
            [
                '--data_dir', data_dir, '--output_dir', output_dir,
                '--embedding_file', embedding_file,
                '--config_file', config_file, '-b', str(batch_size),
                '--lr', str(lr), '-e', str(epochs), '--model_type', tagger_type, '--best_result_file', best_dev_file, '--max_sentence_length',
                '50', '--max_word_length', '12', '--logging_steps', str(logging_steps), '--save_steps', '0',
                '--word_dropout', '0', '--ignore_token=' + ignore_token, '--train_filename', train_filename,
                '--overwrite_output_dir'
            ])
    tagger_procedure.run_procedure(tagger_args)

def train_idcnn_soft():
    tagger_type = 'id-cnn'
    data_dir = "/home/shira_nlp/bert/data/CoNLL"
    precentage_str = '1'
    output_dir = '/home/shira_nlp/idcnn_models' + os.sep + precentage_str
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # embedding_file = '/home/peter_nlp/data/glove.6B/glove.6B.100d.txt'
    embedding_file = '/home/shira_nlp/dilated-cnn-ner/data/embeddings/lample-embeddings-pre_orig.txt'
    config_file = '/home/shira_nlp/idcnn_experiments/config_files/replicate_config.json'
    batch_size = 8
    lr = 4e-5
    epochs = 300
    best_dev_file = '/home/shira_nlp/logs/idcnn_soft_{}.log'.format(precentage_str)
    logging_steps = 300
    ignore_token = "-DOCSTART-"
    train_filename = 'train.txt'

    train_tagger(tagger_type, data_dir, output_dir, embedding_file, config_file, batch_size, lr, epochs,
                    best_dev_file, logging_steps, ignore_token, train_filename)



def train_pseudo_distilled_ner(
        batch_size, train_file, teacher_path, log_file, log_steps, epochs, output_path, lr, unlabeled_file):
    data_dir = "/home/shira_nlp/bert/data/CoNLL"
    # bert_output_path = '/home/shira_nlp/bert_trained_models'
    kd_ps_parser = argparse.ArgumentParser()
    train_kd_ps_procedure = TrainTaggerKDPseudo()
    train_kd_ps_procedure.add_arguments(kd_ps_parser)
    # embeddings_path =  '/home/peter_nlp/data/glove.6B/glove.6B.100d.txt'
    embeddings_path = '/home/shira_nlp/dilated-cnn-ner/data/embeddings/' \
        'lample-embeddings-pre_orig.txt'
    # embeddings_path = None
    # idcnn_output_path = '/home/shira_nlp/idcnn_distilled_models'
    # log_path = '/home/shira_nlp/logs/train_idcnn_log.txt'
    # teacher_path = bert_output_path + '/best_dev'
    teacher_type = 'bert'
    # idcnn
    idcnn_distil_ps_args = kd_ps_parser.parse_args(
        [
            '--data_dir', data_dir, '--output_dir', output_path,
            '--embedding_file', embeddings_path,
            '--config_file',
            '/home/shira_nlp/idcnn_experiments/config_files/replicate_config.json',
            '-b', str(batch_size), '--lr', str(lr), '-e', str(epochs), '--model_type', 'id-cnn',
            '--best_result_file', log_file, '--max_sentence_length', '50', '--max_word_length',
            '12', '--logging_steps', str(log_steps), '--save_steps', '0',
            '--word_dropout', '0', '--ignore_token=-DOCSTART-', '--teacher_model_path',
            teacher_path, '--teacher_model_type', teacher_type, '--kd_loss_fn', 'mse',
            '--unlabeled_filename', unlabeled_file, '--train_filename', train_file,
            '--overwrite_output_dir'
        ])

    train_kd_ps_procedure.run_procedure(idcnn_distil_ps_args)


def train_distilled_ner(
        batch_size, train_file, teacher_path, log_file, log_steps, epochs, output_path, lr):
    data_dir = "/home/shira_nlp/bert/data/CoNLL"
    kd_parser = argparse.ArgumentParser()
    train_kd_procedure = TrainTaggerKD()
    train_kd_procedure.add_arguments(kd_parser)
    # embeddings_path =  '/home/peter_nlp/data/glove.6B/glove.6B.100d.txt'
    embeddings_path = '/home/shira_nlp/dilated-cnn-ner/data/embeddings/' \
        'lample-embeddings-pre_orig.txt'
    # idcnn_output_path = '/home/shira_nlp/idcnn_distilled_models'
    # log_path = '/home/shira_nlp/logs/train_idcnn_log.txt'
    teacher_type = 'bert'
    # idcnn
    idcnn_distil_args = kd_parser.parse_args(
        [
            '--data_dir', data_dir, '--output_dir', output_path,
            '--embedding_file', embeddings_path, '--config_file',
            '/home/shira_nlp/idcnn_experiments/config_files/replicate_config.json',
            '-b', str(batch_size), '--lr', str(lr), '-e', str(epochs), '--model_type',
            'id-cnn', '--best_result_file', log_file, '--max_sentence_length', '50', '--max_word_length',
            '12', '--logging_steps', str(log_steps), '--save_steps', '0',
            '--word_dropout', '0.85', '--ignore_token=-DOCSTART-', '--teacher_model_path',
            teacher_path, '--teacher_model_type', teacher_type, '--kd_loss_fn', 'mse',
            '--train_filename', train_file, '--overwrite_output_dir'
        ])

    train_kd_procedure.run_procedure(idcnn_distil_args)


def train_bert(batch_size, train_file, log_file, log_steps, epochs, bert_output_path, lr):
    data_dir = "/home/shira_nlp/bert/data/CoNLL"
    bert_tag_parser = argparse.ArgumentParser()
    bert_train_procedure = TransformerTokenClsTrain()
    bert_train_procedure.add_arguments(bert_tag_parser)
    model_type = 'bert'
    bert_tag_args = bert_tag_parser.parse_args(
        [
            '--data_dir', data_dir, '--model_name_or_path', 'bert-base-cased', '--model_type', model_type,
            '--output_dir', bert_output_path, '--num_train_epoch', str(epochs), '--max_grad_norm', '5',
            '--learning_rate', str(lr), '--per_gpu_train_batch_size', str(batch_size), '--per_gpu_eval_batch_size', str(batch_size),
            '--logging_steps', str(log_steps), '--ignore_token=-DOCSTART-', '--train_file_name', train_file, '--best_dev_file', log_file,
            '--save_steps', '0',
            '--overwrite_output_dir'
        ])
    # train bert
    bert_train_procedure.run_procedure(bert_tag_args)  


def evaluate(procedure_args):
    # evaluate saved model
    model_type = 'bert'
    device, n_gpus = setup_backend(procedure_args.no_cuda)
    # Set seed
    procedure_args.seed = set_seed(procedure_args.seed, n_gpus)
    bert_output_path = '/home/shira_nlp/bert_trained_models'
    model_path = os.path.join(bert_output_path, 'best_dev')

    bert_trained = TransformerTokenClassifier.load_model(model_path=model_path,
                                                        model_type=model_type,
                                                        config_name=procedure_args.config_name,
                                                            tokenizer_name=procedure_args.tokenizer_name,
                                                            do_lower_case=procedure_args.do_lower_case,
                                                            output_path=procedure_args.output_dir,
                                                            device=device,
                                                            n_gpus=n_gpus,
                                                            bilou=procedure_args.bilou)


    processor = TokenClsProcessor(procedure_args.data_dir, ignore_token=procedure_args.ignore_token)
    dev_ex = processor.get_dev_examples()
    test_ex = processor.get_test_examples()
    if dev_ex is not None:
        dev_dataset = bert_trained.convert_to_tensors(dev_ex,
                                                    max_seq_length=procedure_args.max_seq_length)
        dev_sampler = SequentialSampler(dev_dataset)
        dev_dl = DataLoader(dev_dataset, sampler=dev_sampler,
                            batch_size=procedure_args.per_gpu_eval_batch_size)

    if test_ex is not None:
        test_dataset = bert_trained.convert_to_tensors(test_ex,
                                                        max_seq_length=procedure_args.max_seq_length)
        test_sampler = SequentialSampler(test_dataset)
        test_dl = DataLoader(test_dataset, sampler=test_sampler,
                                batch_size=procedure_args.per_gpu_eval_batch_size)

    bert_trained.model.eval()
    logits, label_ids = bert_trained._evaluate(dev_dl)
    f1 = bert_trained.evaluate_predictions(logits, label_ids)
    print('\n\nf1 dev=' + str(f1) + '\n')
    logits, label_ids = bert_trained._evaluate(test_dl)
    f1 = bert_trained.evaluate_predictions(logits, label_ids)
    print('\n\nf1 test=' + str(f1) + '\n')


def run_ner_distil_experiments(with_unlabeled=False):
    """for each experiment, train bert into a folder, then run distillation
    and write best results into a file"""
    data_dir = "/home/shira_nlp/bert/data/CoNLL"
    params = {
        "labeled_precentage": [0.01, 0.02, 0.05, 0.1, 0.2],
        "batch_size": [8],
        "learning_rate": [4e-5]
        }
    runs_count = 1
    splits_dir = data_dir + os.path.sep + "data_splits_bilou"
    conll_count = 14987
    experiment_folder = '/home/shira_nlp/distil_experiments_ner'
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
        os.makedirs(experiment_folder + '/bert_models')
        os.makedirs(experiment_folder + '/distil_models')
    for p in ParameterGrid(params):
        precentage = p['labeled_precentage']
        precentage_str = str(p['labeled_precentage'])
        batch_size = p['batch_size']
        lr = p['learning_rate']
        split_directory = splits_dir + os.sep + precentage_str
        # if os.path.exists(split_directory):
        if True:
            bert_output_path = experiment_folder + '/bert_models' + '/bert_{}_{}_{}'.format(precentage_str,str(batch_size),str(lr))
            model_output_path = experiment_folder + '/distil_models' + '/idcnn_{}_{}_{}'.format(precentage_str,str(batch_size),str(lr))
            if not os.path.exists(bert_output_path):
                os.makedirs(bert_output_path)
            if not os.path.exists(model_output_path):
                os.makedirs(model_output_path)
            
            for exp in range(runs_count):
                # prepare train file
                distil_train_file = data_dir + os.sep + 'distil_train.txt'
                if os.path.exists(distil_train_file):
                    os.remove(distil_train_file) # remove the one from last experiment and copy the new one
                src_file = splits_dir + os.sep + precentage_str + os.sep + str(exp+1) + '/labeled.txt'
                copyfile(src_file, distil_train_file)
                assert count_examples(distil_train_file) == math.ceil(conll_count * p['labeled_precentage'])

                # prepare output folders for bert/distilled model
                bert_exp_path = bert_output_path + os.sep + str(exp+1)
                model_exp_path = model_output_path + os.sep + str(exp+1)
                bert_log_path = '/home/shira_nlp/logs/train_bert_{}_{}_{}_{}.log'.format(precentage_str, str(batch_size),str(lr),str(exp+1))
                model_log_path = '/home/shira_nlp/logs/train_idcnn_{}_{}_{}_{}.log'.format(precentage_str, str(batch_size),str(lr),str(exp+1))
                if not os.path.exists(bert_exp_path):
                    os.makedirs(bert_exp_path)
                if not os.path.exists(model_exp_path):
                    os.makedirs(model_exp_path)
                # # train bert
                # epochs = 100
                # log_steps = 100
                # train_bert(batch_size, 'distil_train.txt', bert_log_path, log_steps, epochs, bert_exp_path, lr)
                teacher_path = bert_exp_path + '/best_dev'
                epochs = 300
                log_steps = 1000
                if with_unlabeled:
                    # batch_size = 32
                    # lr = 0.03
                    train_pseudo_distilled_ner(
                        batch_size, 'distil_train.txt', teacher_path, model_log_path, log_steps, epochs, model_exp_path, lr, '10000_unlabeled_bilou.txt')
                else:
                    train_distilled_ner(batch_size, 'distil_train.txt', teacher_path, model_log_path, log_steps, epochs, model_exp_path, lr)


# run_ner_distil_experiments(with_unlabeled=True)
train_idcnn_soft()