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

"""
This example uses the Amazon reviews though additional datasets can easily be substituted.
It only requires text and a sentiment label
It then takes the dataset and trains two models (again can be expanded)
The labels for the test data is then predicted.
The same train and test data is used for both models

The ensembler takes the two prediction matrixes and weights (as defined by model accuracy)
and determines the final prediction matrix.

Finally, the full classification report is displayed.

A similar pipeline could be utilized to train models on a dataset, predict on a second dataset
and aquire a list of final predictions
"""

import argparse

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
# pylint: disable=no-name-in-module
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer

from nlp_architect.data.amazon_reviews import Amazon_Reviews
from nlp_architect.models.supervised_sentiment import simple_lstm, one_hot_cnn
from nlp_architect.utils.ensembler import simple_ensembler
from nlp_architect.utils.generic import to_one_hot
from nlp_architect.utils.io import validate_existing_filepath, check_size

max_fatures = 2000
max_len = 300
batch_size = 32
embed_dim = 256
lstm_out = 140


def ensemble_models(data, args):
    # split, train, test
    data.process()
    dense_out = len(data.labels[0])
    # split for all models
    X_train_, X_test_, Y_train, Y_test = train_test_split(data.text, data.labels,
                                                          test_size=0.20, random_state=42)

    # Prep data for the LSTM model
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(X_train_)
    X_train = tokenizer.texts_to_sequences(X_train_)
    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = tokenizer.texts_to_sequences(X_test_)
    X_test = pad_sequences(X_test, maxlen=max_len)

    # Train the LSTM model
    lstm_model = simple_lstm(max_fatures, dense_out, X_train.shape[1], embed_dim, lstm_out)
    model_hist = lstm_model.fit(X_train, Y_train, epochs=args.epochs, batch_size=batch_size,
                                verbose=1, validation_data=(X_test, Y_test))
    lstm_acc = model_hist.history['acc'][-1]
    print("LSTM model accuracy ", lstm_acc)

    # And make predictions using the LSTM model
    lstm_predictions = lstm_model.predict(X_test)

    # Now prep data for the one-hot CNN model
    X_train_cnn = np.asarray([to_one_hot(x) for x in X_train_])
    X_test_cnn = np.asarray([to_one_hot(x) for x in X_test_])

    # And train the one-hot CNN classifier
    model_cnn = one_hot_cnn(dense_out, max_len)
    model_hist_cnn = model_cnn.fit(X_train_cnn, Y_train, batch_size=batch_size, epochs=args.epochs,
                                   verbose=1, validation_data=(X_test_cnn, Y_test))
    cnn_acc = model_hist_cnn.history['acc'][-1]
    print("CNN model accuracy: ", cnn_acc)

    # And make predictions
    one_hot_cnn_predictions = model_cnn.predict(X_test_cnn)

    # Using the accuracies create an ensemble
    accuracies = [lstm_acc, cnn_acc]
    norm_accuracies = [a / sum(accuracies) for a in accuracies]

    print("Ensembling with weights: ")
    for na in norm_accuracies:
        print(na)
    ensembled_predictions = simple_ensembler([lstm_predictions, one_hot_cnn_predictions],
                                             norm_accuracies)
    final_preds = np.argmax(ensembled_predictions, axis=1)

    # Get the final accuracy
    print(classification_report(np.argmax(Y_test, axis=1), final_preds,
                                target_names=data.labels_0.columns.values))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='./',
                        help='file_path where the files to parse are located')
    parser.add_argument('--data_type', type=str, default='amazon',
                        choices=['amazon'],
                        help='dataset source')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for both models', action=check_size(1, 20000))
    args_in = parser.parse_args()

    # Check file path
    if args_in.file_path:
        validate_existing_filepath(args_in.file_path)

    if args_in.data_type == 'amazon':
        data_in = Amazon_Reviews(args_in.file_path)
    ensemble_models(data_in, args_in)
