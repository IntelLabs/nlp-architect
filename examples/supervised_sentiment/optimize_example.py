# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

import pickle
import argparse
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp, Trials
# pylint: disable=no-name-in-module
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer

from nlp_architect.data.amazon_reviews import Amazon_Reviews
from nlp_architect.models.supervised_sentiment import simple_lstm
from nlp_architect.utils.io import validate_parent_exists, check_size, validate_existing_filepath

max_len = 100
batch_size = 32


def run_loss(args):
    data = args['data']

    # For each run we want to get a new random balance
    data.process()
    # split, train, test
    dense_out = len(data.labels[0])
    # split for all models
    X_train_, X_test_, Y_train, Y_test = train_test_split(data.text, data.labels,
                                                          test_size=0.20, random_state=42)

    print(args)

    # Prep data for the LSTM model
    # This currently will train the tokenizer on all text (unbalanced and train/test)
    # It would be nice to replace this with a pretrained embedding on larger text

    tokenizer = Tokenizer(num_words=int(args['max_features']), split=' ')
    tokenizer.fit_on_texts(data.all_text)
    X_train = tokenizer.texts_to_sequences(X_train_)
    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = tokenizer.texts_to_sequences(X_test_)
    X_test = pad_sequences(X_test, maxlen=max_len)

    # Train the LSTM model
    lstm_model = simple_lstm(int(args['max_features']), dense_out, X_train.shape[1],
                             int(args['embed_dim']), int(args['lstm_out']), args['dropout'])

    if args['epochs'] == 0:
        args['epochs'] = 1

    es = EarlyStopping(monitor='val_acc', min_delta=0, patience=6, verbose=0, mode='max')
    model_hist = lstm_model.fit(X_train, Y_train, epochs=args['epochs'], batch_size=batch_size,
                                verbose=1, validation_data=(X_test, Y_test), callbacks=[es])
    lstm_acc = model_hist.history['val_acc'][-1]
    print("LSTM model accuracy ", lstm_acc)
    # This minimizes, so the maximize we have to take the inverse :)
    return 1 - lstm_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=validate_existing_filepath, default='./',
                        help='file_path where the files to parse are located')
    parser.add_argument('--data_type', type=str, default='amazon',
                        choices=['amazon'])
    parser.add_argument('--output_file', type=validate_parent_exists, default='./opt_trials.pkl',
                        help='file_path where the output of the trials will be located')
    parser.add_argument('--new_trials', type=int, default=20, action=check_size(1, 20000))
    args_in = parser.parse_args()

    # Check inputs
    if args_in.file_path:
        validate_existing_filepath(args_in.file_path)
    if args_in.output_file:
        validate_parent_exists(args_in.output_file)

    if args_in.data_type == 'amazon':
        data_in = Amazon_Reviews(args_in.file_path)

    try:
        if args_in.output_file.endswith('.pkl'):
            with open(args_in.output_file, 'rb') as read_f:
                trials_to_keep = pickle.load(read_f)
            print("Utilizing existing trial files")
        else:
            trials_to_keep = Trials()
    # If the file does not already exist we will start with a new set of trials
    except FileNotFoundError:
        trials_to_keep = Trials()

    space = {'data': data_in,
             'max_features': hp.choice('max_features', [500, 1000, 2000, 3000]),
             'embed_dim': hp.uniform('embed_dim', 100, 500),
             'lstm_out': hp.uniform('lstm_out', 50, 300),
             'epochs': hp.randint('epochs', 50),
             'dropout': hp.uniform('dropout', 0, 0.1)
             }

    num_evals = len(trials_to_keep.trials) + args_in.new_trials
    best = fmin(run_loss,
                space=space,
                algo=tpe.suggest,
                max_evals=num_evals,
                trials=trials_to_keep
                )
    # Write out the trials
    with open(args_in.output_file, 'wb') as f:
        pickle.dump(trials_to_keep, f)
