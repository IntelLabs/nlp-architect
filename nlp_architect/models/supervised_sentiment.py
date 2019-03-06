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
import tensorflow as tf


def simple_lstm(max_features, dense_out, input_length, embed_dim=256, lstm_out=140,
                dropout=0.5):
    """
    Simple Bi-direction LSTM Model in Keras

    Single layer bi-directional lstm with recurrent dropout and a fully connected layer

    Args:
        max_features (int): vocabulary size
        dense_out (int): size out the output dense layer, this is the number of classes
        input_length (int): length of the input text
        embed_dim (int): internal embedding size used in the lstm
        lstm_out (int): size of the bi-directional output layer
        dropout (float): value for recurrent dropout, between 0 and 1

    Returns:
        model (model): LSTM model
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(max_features, embed_dim, input_length=input_length))
    model.add(tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_out, recurrent_dropout=dropout, activation='tanh')))
    model.add(tf.keras.layers.Dense(dense_out, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def one_hot_cnn(dense_out, max_len=300, frame='small'):
    """
    Temporal CNN Model

    As defined in "Text Understanding from Scratch" by Zhang, LeCun 2015
    https://arxiv.org/pdf/1502.01710v4.pdf
    This model is a series of 1D CNNs, with a maxpooling and fully connected layers.
    The frame sizes may either be large or small.


    Args:
        dense_out (int): size out the output dense layer, this is the number of classes
        max_len (int): length of the input text
        frame (str): frame size, either large or small

    Returns:
        model (model): temporal CNN model
    """

    if frame == 'large':
        cnn_size = 1024
        fully_connected = [2048, 2048, dense_out]
    else:
        cnn_size = 256
        fully_connected = [1024, 1024, dense_out]

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv1D(cnn_size, 7, padding='same', input_shape=(68, max_len)))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=3))

    print(model.output_shape)

    # Input = 22 x 256
    model.add(tf.keras.layers.Conv1D(cnn_size, 7, padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=3))

    print(model.output_shape)
    # Input = 7 x 256
    model.add(tf.keras.layers.Conv1D(cnn_size, 3, padding='same'))

    # Input = 7 x 256
    model.add(tf.keras.layers.Conv1D(cnn_size, 3, padding='same'))

    model.add(tf.keras.layers.Conv1D(cnn_size, 3, padding='same'))

    # Input = 7 x 256
    model.add(tf.keras.layers.Conv1D(cnn_size, 3, padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=3))

    model.add(tf.keras.layers.Flatten())

    # Fully Connected Layers

    # Input is 512 Output is 1024/2048
    model.add(tf.keras.layers.Dense(fully_connected[0]))
    model.add(tf.keras.layers.Dropout(0.75))
    model.add(tf.keras.layers.Activation('relu'))

    # Input is 1024/2048 Output is 1024/2048
    model.add(tf.keras.layers.Dense(fully_connected[1]))
    model.add(tf.keras.layers.Dropout(0.75))
    model.add(tf.keras.layers.Activation('relu'))

    # Input is 1024/2048 Output is dense_out size (number of classes)
    model.add(tf.keras.layers.Dense(fully_connected[2]))
    model.add(tf.keras.layers.Activation('softmax'))

    # Stochastic gradient parameters as set by paper
    sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model
