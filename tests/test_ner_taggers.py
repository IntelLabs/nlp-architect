import argparse
import os
import pytest
import tempfile
import shutil
from nlp_architect.procedures import TrainTagger
from nlp_architect.nn.torch.modules.embedders import IDCNN, CNNLSTM
import torch

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'fixtures/conll_sample')
output_dir = tempfile.mkdtemp()
parser = argparse.ArgumentParser()
train_procedure = TrainTagger()
train_procedure.add_arguments(parser)
embeddings_path = None
batch_size = 128
learning_rate = 0.0008
epochs = 1
train_filename = 'data.txt'


def test_taggers():
    words= torch.tensor([[1,2,3,4,5,0,0,0]], dtype=torch.long) # (1,8)
    word_chars= torch.tensor([[[1,2,0],
                            [3,0,0],
                            [4,5,0],
                            [5,4,3],
                            [2,2,0],
                            [0,0,0],
                            [0,0,0],
                            [0,0,0]]], dtype=torch.long) # (1, 8, 3)
    shapes= torch.tensor([[1,2,3,3,3,0,0,0]], dtype=torch.long) # (1,8)
    mask= torch.tensor([[1,1,1,1,1,0,0,0]], dtype=torch.long) # (1,8)
    labels= torch.tensor([[1,1,3,4,1,0,0,0]], dtype=torch.long) # (1,8)

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
    lstm_logits = idcnn_model(**inputs)
    assert lstm_logits.shape == expected_output_shape


def test_tagging_procedure_sanity():

    # run idcnn softmax
    model_type = 'id-cnn'
    idcnn_softmax_args = parser.parse_args(
            [
                '--data_dir', data_dir, '--output_dir', output_dir,
                '--embedding_file', embeddings_path, '-b', str(batch_size),
                '--lr', str(learning_rate), '-e', str(epochs), '--train_filename', train_filename,
                '--dev_filename', train_filename, '--test_filename', train_filename, '--model_type', model_type,
                '--overwrite_output_dir'
            ])
    train_procedure.run_procedure(idcnn_softmax_args)

    # run idcnn crf
    idcnn_crf_args = parser.parse_args(
            [
                '--data_dir', data_dir, '--output_dir', output_dir,
                '--embedding_file', embeddings_path, '-b', str(batch_size),
                '--lr', str(learning_rate), '-e', str(epochs), '--train_filename', train_filename,
                '--dev_filename', train_filename, '--test_filename', train_filename, '--model_type', model_type,
                '--use_crf', '--overwrite_output_dir'
            ])
    train_procedure.run_procedure(idcnn_crf_args)


    # run lstm softmax 
    model_type = 'cnn-lstm'
    lstm_softmax_args = parser.parse_args(
            [
                '--data_dir', data_dir, '--output_dir', output_dir,
                '--embedding_file', embeddings_path, '-b', str(batch_size),
                '--lr', str(learning_rate), '-e', str(epochs), '--train_filename', train_filename,
                '--dev_filename', train_filename, '--test_filename', train_filename, '--model_type', model_type,
                '--overwrite_output_dir'
            ])
    train_procedure.run_procedure(lstm_softmax_args)    

    # run lstm crf 
    lstm_crf_args = parser.parse_args(
            [
                '--data_dir', data_dir, '--output_dir', output_dir,
                '--embedding_file', embeddings_path, '-b', str(batch_size),
                '--lr', str(learning_rate), '-e', str(epochs), '--train_filename', train_filename,
                '--dev_filename', train_filename, '--test_filename', train_filename, '--model_type', model_type,
                '--use_crf', '--overwrite_output_dir'
            ])
    train_procedure.run_procedure(lstm_crf_args)    

    # remove output files
    shutil.rmtree(output_dir)

    assert True