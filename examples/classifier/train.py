# -*- coding: utf-8 -*-

"""
Created on 2018/10/7 上午1:17

@author: xujiang@baixing.com

"""

from os import path
import argparse

from nlp_architect.utils.io import validate_existing_filepath, validate_parent_exists, validate

def read_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=10,
                        help='Batch size')
    parser.add_argument('-e', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--train_file', type=validate_existing_filepath, required=True,
                        help='Train file (sequential tagging dataset format)')
    parser.add_argument('--test_file', type=validate_existing_filepath, required=True,
                        help='Test file (sequential tagging dataset format)')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='classifier labels number in train/test files')
    parser.add_argument('--seq_length', type=int, default=600,
                        help='序列长度')
    parser.add_argument('--num_filters', type=int, default=256,
                        help='卷积核数量')
    parser.add_argument('--kernel_size', type=int, default=5,
                        help='卷积核尺寸')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='全连接层神经元')
    parser.add_argument('--character_embedding_dims', type=int, default=300,
                        help='Character features embedding dimension size')
    parser.add_argument('--char_features_lstm_dims', type=int, default=300,
                        help='Character feature extractor LSTM dimension size')
    parser.add_argument('--entity_tagger_lstm_dims', type=int, default=100,
                        help='Entity tagger LSTM dimension size')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--embedding_model', type=validate_existing_filepath,
                        help='Path to external word embedding model file')
    parser.add_argument('--model_path', type=str, default='model.h5',
                        help='Path for saving model weights')
    parser.add_argument('--model_info_path', type=str, default='model_info.dat',
                        help='Path for saving model topology')
    input_args = parser.parse_args()
    validate_input_args(input_args)
    return input_args

def validate_input_args(args):
    validate((args.b, int, 1, 100000))
    validate((args.e, int, 1, 100000))
    validate((args.tag_num, int, 1, 1000))
    validate((args.sentence_length, int, 1, 10000))
    validate((args.word_length, int, 1, 100))
    validate((args.word_embedding_dims, int, 1, 10000))
    validate((args.character_embedding_dims, int, 1, 1000))
    validate((args.char_features_lstm_dims, int, 1, 10000))
    validate((args.entity_tagger_lstm_dims, int, 1, 10000))
    validate((args.dropout, float, 0, 1))
    model_path = path.join(path.dirname(path.realpath(__file__)), str(args.model_path))
    validate_parent_exists(model_path)
    model_info_path = path.join(path.dirname(path.realpath(__file__)), str(args.model_info_path))
    validate_parent_exists(model_info_path)
