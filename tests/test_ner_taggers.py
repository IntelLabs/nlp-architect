import argparse
import os
import tempfile
import shutil
import torch
import pytest
from nlp_architect.procedures import TrainTagger
from nlp_architect.nn.torch.modules.embedders import IDCNN, CNNLSTM


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'fixtures/conll_sample')
OUTPUT_DIR = tempfile.mkdtemp()
PARSER = argparse.ArgumentParser()
TRAIN_PROCEDURE = TrainTagger()
TRAIN_PROCEDURE.add_arguments(PARSER)
EMBEDDINGS_PATH = None
BATCH_SIZE = 128
LEARNING_RATE = 0.0008
EPOCHS = 1
TRAIN_FILENAME = 'data.txt'


def test_taggers():
    words = torch.tensor([[1,2,3,4,5,0,0,0]], dtype=torch.long) # (1,8)
    word_chars = torch.tensor([[[1,2,0],
                            [3,0,0],
                            [4,5,0],
                            [5,4,3],
                            [2,2,0],
                            [0,0,0],
                            [0,0,0],
                            [0,0,0]]], dtype=torch.long) # (1, 8, 3)
    shapes = torch.tensor([[1,2,3,3,3,0,0,0]], dtype=torch.long) # (1,8)
    mask = torch.tensor([[1,1,1,1,1,0,0,0]], dtype=torch.long) # (1,8)
    labels = torch.tensor([[1,1,3,4,1,0,0,0]], dtype=torch.long) # (1,8)

    word_vocab_size = 5
    label_vocab_size = 4

    inputs = {'words': words,
            'word_chars': word_chars,
            'shapes': shapes,
            'mask': mask,
            'labels': labels}

    idcnn_model = IDCNN(word_vocab_size + 1, label_vocab_size + 1)
    lstm_model = CNNLSTM(word_vocab_size + 1, label_vocab_size + 1)
    expected_output_shape = torch.Size([1, 8, label_vocab_size + 1])

    idcnn_logits = idcnn_model(**inputs)
    assert idcnn_logits.shape == expected_output_shape
    lstm_logits = lstm_model(**inputs)
    assert lstm_logits.shape == expected_output_shape


def test_tagging_procedure_sanity():

    # run idcnn softmax
    model_type = 'id-cnn'
    idcnn_softmax_args = PARSER.parse_args(
            [
                '--data_dir', DATA_DIR, '--output_dir', OUTPUT_DIR,
                '--embedding_file', EMBEDDINGS_PATH, '-b', str(BATCH_SIZE),
                '--lr', str(LEARNING_RATE), '-e', str(EPOCHS), '--train_filename', TRAIN_FILENAME,
                '--dev_filename', TRAIN_FILENAME, '--test_filename', TRAIN_FILENAME, '--model_type', model_type,
                '--overwrite_output_dir'
            ])
    TRAIN_PROCEDURE.run_procedure(idcnn_softmax_args)

    # run idcnn crf
    idcnn_crf_args = PARSER.parse_args(
            [
                '--data_dir', DATA_DIR, '--output_dir', OUTPUT_DIR,
                '--embedding_file', EMBEDDINGS_PATH, '-b', str(BATCH_SIZE),
                '--lr', str(LEARNING_RATE), '-e', str(EPOCHS), '--train_filename', TRAIN_FILENAME,
                '--dev_filename', TRAIN_FILENAME, '--test_filename', TRAIN_FILENAME, '--model_type', model_type,
                '--use_crf', '--overwrite_output_dir'
            ])
    TRAIN_PROCEDURE.run_procedure(idcnn_crf_args)


    # run lstm softmax 
    model_type = 'cnn-lstm'
    lstm_softmax_args = PARSER.parse_args(
            [
                '--data_dir', DATA_DIR, '--output_dir', OUTPUT_DIR,
                '--embedding_file', EMBEDDINGS_PATH, '-b', str(BATCH_SIZE),
                '--lr', str(LEARNING_RATE), '-e', str(EPOCHS), '--train_filename', TRAIN_FILENAME,
                '--dev_filename', TRAIN_FILENAME, '--test_filename', TRAIN_FILENAME, '--model_type', model_type,
                '--overwrite_output_dir'
            ])
    TRAIN_PROCEDURE.run_procedure(lstm_softmax_args)    

    # run lstm crf 
    lstm_crf_args = PARSER.parse_args(
            [
                '--data_dir', DATA_DIR, '--output_dir', OUTPUT_DIR,
                '--embedding_file', EMBEDDINGS_PATH, '-b', str(BATCH_SIZE),
                '--lr', str(LEARNING_RATE), '-e', str(EPOCHS), '--train_filename', TRAIN_FILENAME,
                '--dev_filename', TRAIN_FILENAME, '--test_filename', TRAIN_FILENAME, '--model_type', model_type,
                '--use_crf', '--overwrite_output_dir'
            ])
    TRAIN_PROCEDURE.run_procedure(lstm_crf_args)    

    # remove output files
    shutil.rmtree(OUTPUT_DIR)
    assert True
