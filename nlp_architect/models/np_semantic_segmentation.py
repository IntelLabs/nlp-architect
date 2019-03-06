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


# taken from keras previous versions: https://github.com/keras-team/keras/issues/5400
def precision_score(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    K = tf.keras.backend
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall_score(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    K = tf.keras.backend
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred):
    """

    Args:
        y_true:
        y_pred:

    Returns:

    """
    K = tf.keras.backend
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


class NpSemanticSegClassifier:
    """
    NP Semantic Segmentation classifier model (based on tf.Keras framework).

    Args:
        num_epochs(int): number of epochs to train the model
        **callback_args (dict): callback args keyword arguments to init a Callback for the model
        loss: the model's cost function. Default is 'tf.keras.losses.binary_crossentropy' loss
        optimizer (:obj:`tf.keras.optimizers`): the model's optimizer. Default is 'adam'
    """

    def __init__(self, num_epochs, callback_args, loss='binary_crossentropy', optimizer='adam',
                 batch_size=128, ):
        """
        Args:
            num_epochs(int): number of epochs to train the model
            callback_args (dict): callback args keyword arguments to init Callback for the model
            loss: the model's loss function. Default is 'tf.keras.losses.binary_crossentropy' loss
            optimizer (:obj:`tf.keras.optimizers`): the model's optimizer. Default is `adam`
            batch_size (int):  batch size
        """
        self.model = None
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = num_epochs
        self.callback_args = callback_args
        self.batch_size = batch_size

    def build(self, input_dim):
        """
        Build the model's layers
        Args:
            input_dim (int): the first layer's input_dim
        """
        first_layer_dens = 64
        second_layer_dens = 64
        output_layer_dens = 1
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(first_layer_dens, activation='relu', input_dim=input_dim))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(second_layer_dens, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(output_layer_dens, activation='sigmoid'))
        metrics = ['binary_accuracy', precision_score, recall_score, f1]
        # Compile model
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=metrics)
        self.model = model

    def fit(self, train_set):
        """
        Train and fit the model on the datasets

        Args:
            train_set (:obj:`numpy.ndarray`): The train set
            args: callback_args and epochs from ArgParser input
        """
        self.model.fit(train_set['X'], train_set['y'], epochs=self.epochs,
                       batch_size=self.batch_size, verbose=2)

    def save(self, model_path):
        """
        Save the model's prm file in model_path location

        Args:
            model_path(str): local path for saving the model
        """
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(model_path[:-2] + 'json', 'w') as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(model_path)
        print("Saved model to disk")

    def load(self, model_path):
        """
        Load pre-trained model's .h5 file to NpSemanticSegClassifier object

        Args:
            model_path(str): local path for loading the model
        """
        # load json and create model
        with open(model_path[:-2] + 'json', 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = tf.keras.models.model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(model_path)
        print("Loaded model from disk")
        self.model = loaded_model

    def eval(self, test_set):
        """
        Evaluate the model's test_set on error_rate, test_accuracy_rate and precision_recall_rate

        Args:
            test_set (:obj:`numpy.ndarray`): The test set

        Returns:
            tuple(float): loss, binary_accuracy, precision, recall and f1 measures
        """
        return self.model.evaluate(test_set['X'], test_set['y'], batch_size=128, verbose=2)

    def get_outputs(self, test_set):
        """
        Classify the dataset on the model

        Args:
            test_set (:obj:`numpy.ndarray`): The test set

        Returns:
            list(:obj:`numpy.ndarray`): model's predictions
        """
        return self.model.predict(test_set)
