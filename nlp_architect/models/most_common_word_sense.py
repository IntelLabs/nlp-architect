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
# ****************************************************************************

from neon.callbacks import Callbacks
from neon.initializers import Gaussian
from neon.layers import Affine, GeneralizedCost
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import SumSquared, Softmax, Rectlin
from neon.transforms import Misclassification


class MostCommonWordSense:

    def __init__(self, rounding, callback_args, epochs):
        # setup weight initialization function
        self.init = Gaussian(loc=0.0, scale=0.01)
        # setup optimizer
        self.optimizer = GradientDescentMomentum(learning_rate=0.1, momentum_coef=0.9,
                                                 stochastic_round=rounding)
        # setup cost function as CrossEntropy
        self.cost = GeneralizedCost(costfunc=SumSquared())
        self.epochs = epochs
        self.model = None
        self.callback_args = callback_args

    def build(self):
        # setup model layers
        layers = [Affine(nout=100, init=self.init, bias=self.init, activation=Rectlin()),
                  Affine(nout=2, init=self.init, bias=self.init, activation=Softmax())]

        # initialize model object
        self.model = Model(layers=layers)

    def fit(self, valid_set, train_set):
        # configure callbacks
        callbacks = Callbacks(self.model, eval_set=valid_set, **self.callback_args)
        self.model.fit(train_set, optimizer=self.optimizer, num_epochs=self.epochs,
                       cost=self.cost, callbacks=callbacks)

    def save(self, save_path):
        self.model.save_params(save_path)

    def load(self, model_path):
        self.model = Model(model_path)

    def eval(self, valid_set):
        eval_rate = self.model.eval(valid_set, metric=Misclassification())
        return eval_rate

    def get_outputs(self, valid_set):
        return self.model.get_outputs(valid_set)
