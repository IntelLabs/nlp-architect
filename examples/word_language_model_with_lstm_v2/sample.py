# -*- coding: utf-8 -*-

"""
Created on 2018/10/10 上午10:44

@author: xujiang@baixing.com

"""

import tensorflow as tf
from data_preprocessing import Tokenizer
from nlp_architect.models.gen_char_rnn import CharRNN
import os

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('n_neurons', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('n_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('embedding', False, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_string('tokenizer_path', '', 'model/model_name/tokenizer.pkl')
tf.flags.DEFINE_string('checkpoint_path', '', 'checkpoint path')
tf.flags.DEFINE_string('start_string', '', 'use this string to start generating')
tf.flags.DEFINE_integer('max_length', 300, 'max length to generate')

FLAGS.checkpoint_path = 'model/Chinese_poetry/'
FLAGS.tokenizer_path = 'model/Chinese_poetry/tokenizer.pkl'
FLAGS.embedding = True
# FLAGS.start_string = '春暖花开日'
FLAGS.start_string = ''

def main(_):
    tokenizer = Tokenizer(vocab_path=FLAGS.tokenizer_path)
    if os.path.isdir(FLAGS.checkpoint_path):
        FLAGS.checkpoint_path =\
            tf.train.latest_checkpoint(FLAGS.checkpoint_path)

    model = CharRNN(tokenizer.vocab_size, sampling=True,
                    n_neurons=FLAGS.n_neurons, n_layers=FLAGS.n_layers,
                    embedding=FLAGS.embedding,
                    embedding_size=FLAGS.embedding_size)

    model.load(FLAGS.checkpoint_path)

    start = tokenizer.texts_to_sequences(FLAGS.start_string)
    arr = model.predict(FLAGS.max_length, start, tokenizer.vocab_size)
    print(tokenizer.sequences_to_texts(arr))


if __name__ == '__main__':
    tf.app.run()
'''
python sample.py --embedding --tokenizer_path model/gcc/tokenizer.pkl --checkpoint_path model/gcc/ --max_length 100 start_string 工程车
python sample.py --tokenizer_path model/jay/tokenizer.pkl --checkpoint_path  model/jay  --max_length 500  --embedding --n_layers 3 --start_string 晴天
'''