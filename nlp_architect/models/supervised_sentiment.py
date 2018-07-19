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

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Activation, Dropout, Flatten, Bidirectional
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.optimizers import SGD


def simple_lstm(max_fatures, dense_out, input_length, embed_dim=256, lstm_out=140,
                dropout=0.5):
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim, input_length=input_length))
    model.add(Bidirectional(LSTM(lstm_out, recurrent_dropout=dropout, activation='tanh')))
    model.add(Dense(dense_out, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def one_hot_cnn(dense_out, max_len=300, frame='small'):

    if frame == 'large':
        cnn_size = 1024
        fully_connected = [2048, 2048, dense_out]
    else:
        cnn_size = 256
        fully_connected = [1024, 1024, dense_out]

    model = Sequential()

    model.add(Conv1D(cnn_size, 7, padding='same', input_shape=(68, max_len)))
    model.add(MaxPooling1D(pool_size=3))

    print(model.output_shape)

    # Input = 22 x 256
    model.add(Conv1D(cnn_size, 7, padding='same'))
    model.add(MaxPooling1D(pool_size=3))

    print(model.output_shape)
    # Input = 7 x 256
    model.add(Conv1D(cnn_size, 3, padding='same'))

    # Input = 7 x 256
    model.add(Conv1D(cnn_size, 3, padding='same'))

    model.add(Conv1D(cnn_size, 3, padding='same'))

    # Input = 7 x 256
    model.add(Conv1D(cnn_size, 3, padding='same'))
    model.add(MaxPooling1D(pool_size=3))

    model.add(Flatten())

    # Fully Connected Layers

    # Input is 512 Output is 1024/2048
    model.add(Dense(fully_connected[0]))
    model.add(Dropout(0.75))
    model.add(Activation('relu'))

    # Input is 1024/2048 Output is 1024/2048
    model.add(Dense(fully_connected[1]))
    model.add(Dropout(0.75))
    model.add(Activation('relu'))

    # Input is 1024/2048 Output is dense_out size (number of classes)
    model.add(Dense(fully_connected[2]))
    model.add(Activation('softmax'))

    # Stochastic gradient parameters as set by paper
    sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model
