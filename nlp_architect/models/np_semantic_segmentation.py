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

from __future__ import unicode_literals, print_function, division, \
    absolute_import

from neon.callbacks.callbacks import Callbacks
from neon.initializers import Gaussian
from neon.layers import GeneralizedCost, Affine
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, Logistic, CrossEntropyBinary, \
    Misclassification, Accuracy, PrecisionRecall


class NpSemanticSegClassifier:
    """
    NP Semantic Segmentation classifier model (based on Neon framework).

    Args:
        num_epochs(int): number of epochs to train the model
        **callback_args (dict): callback args keyword arguments to init a Callback for the model
        cost: the model's cost function. Default is 'neon.transforms.CrossEntropyBinary' cost
        optimizer (:obj:`neon.optimizers`): the model's optimizer. Default is
        'neon.optimizers.GradientDescentMomentum(0.07, momentum_coef=0.9)'
    """

    def __init__(self, num_epochs, callback_args,
                 optimizer=GradientDescentMomentum(0.07, momentum_coef=0.9)):
        """

        Args:
            num_epochs(int): number of epochs to train the model
            **callback_args (dict): callback args keyword arguments to init Callback for the model
            cost: the model's cost function. Default is 'neon.transforms.CrossEntropyBinary' cost
            optimizer (:obj:`neon.optimizers`): the model's optimizer. Default is
            `neon.optimizers.GradientDescentMomentum(0.07, momentum_coef=0.9)`
        """
        self.model = None
        self.cost = GeneralizedCost(costfunc=CrossEntropyBinary())
        self.optimizer = optimizer
        self.epochs = num_epochs
        self.callback_args = callback_args

    def build(self):
        """
        Build the model's layers
        """
        first_layer_dens = 64
        second_layer_dens = 64
        output_layer_dens = 2
        # setup weight initialization function
        init_norm = Gaussian(scale=0.01)
        # setup model layers
        layers = [Affine(nout=first_layer_dens, init=init_norm,
                         activation=Rectlin()),
                  Affine(nout=second_layer_dens, init=init_norm,
                         activation=Rectlin()),
                  Affine(nout=output_layer_dens, init=init_norm,
                         activation=Logistic(shortcut=True))]

        # initialize model object
        self.model = Model(layers=layers)

    def fit(self, test_set, train_set):
        """
        Train and fit the model on the datasets

        Args:
            test_set (:obj:`neon.data.ArrayIterators`): The test set
            train_set (:obj:`neon.data.ArrayIterators`): The train set
            args: callback_args and epochs from ArgParser input
        """
        # configure callbacks
        callbacks = Callbacks(self.model, eval_set=test_set, **self.callback_args)
        self.model.fit(train_set, optimizer=self.optimizer, num_epochs=self.epochs, cost=self.cost,
                       callbacks=callbacks)

    def save(self, model_path):
        """
        Save the model's prm file in model_path location

        Args:
            model_path(str): local path for saving the model
        """
        self.model.save_params(model_path)

    def load(self, model_path):
        """
        Load pre-trained model's .prm file to NpSemanticSegClassifier object

        Args:
            model_path(str): local path for loading the model
        """
        self.model = Model(model_path)

    def eval(self, test_set):
        """
        Evaluate the model's test_set on error_rate, test_accuracy_rate and precision_recall_rate

        Args:
            test_set (ArrayIterator): The test set

        Returns:
            tuple(int): error_rate, test_accuracy_rate and precision_recall_rate
        """
        error_rate = self.model.eval(test_set, metric=Misclassification())
        test_accuracy_rate = self.model.eval(test_set, metric=Accuracy())
        precision_recall_rate = self.model.eval(test_set, metric=PrecisionRecall(2))
        return error_rate, test_accuracy_rate, precision_recall_rate

    def get_outputs(self, test_set):
        """
        Classify the dataset on the model

        Args:
            test_set (:obj:`neon.data.ArrayIterators`): The test set

        Returns:
            list(float): model's predictions
        """
        return self.model.get_outputs(test_set)
