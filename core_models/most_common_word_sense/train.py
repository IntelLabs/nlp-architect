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
"""
Most Common Word Sense - Train MLP classifier and evaluate it.
"""

from neon.backends import gen_backend
from neon.callbacks import Callbacks
from neon.data import ArrayIterator
from neon.initializers import Gaussian
from neon.layers import Affine, GeneralizedCost
from neon.optimizers import GradientDescentMomentum
from neon.transforms import SumSquared, Softmax, Misclassification, Rectlin
from neon.models import Model
from neon.util.argparser import NeonArgparser
import pickle
import logging

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def most_common_word_train(x_train, y_train, x_valid, y_valid):
    # train set
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    train_set = ArrayIterator(X=x_train, y=y_train, make_onehot=False)

    # validation set
    x_valid = np.array(x_valid)
    y_valid = np.array(y_valid)
    valid_set = ArrayIterator(X=x_valid, y=y_valid, make_onehot=False)

    # setup weight initialization function
    init = Gaussian(loc=0.0, scale=0.01)

    # setup model layers
    layers = [Affine(nout=100, init=init, bias=init, activation=Rectlin()),
              Affine(nout=2, init=init, bias=init, activation=Softmax())]

    # initialize model object
    mlp_model = Model(layers=layers)

    # setup optimizer
    optimizer = GradientDescentMomentum(learning_rate=0.1, momentum_coef=0.9, stochastic_round=args.rounding)

    # setup cost function as CrossEntropy
    cost = GeneralizedCost(costfunc=SumSquared())

    # configure callbacks
    callbacks = Callbacks(mlp_model, eval_set=valid_set, **args.callback_args)

    # train
    mlp_model.fit(train_set, optimizer=optimizer, num_epochs=args.epochs, cost=cost, callbacks=callbacks)

    mlp_model.save_params(args.model_prm)

    # evaluation
    error_rate = mlp_model.eval(valid_set, metric=Misclassification())
    logger.info('Misclassification error on validation set= %.1f%%' % (error_rate * 100))

    results = mlp_model.get_outputs(valid_set)

    return results

# -------------------------------------------------------------------------------------#


if __name__ == "__main__":
    # parse the command line arguments
    parser = NeonArgparser(__doc__)
    parser.add_argument('--data_set_file', default='data/data_set.pkl',
                        help='train and validation sets path')
    parser.add_argument('--model_prm', default='data/mcs_model.prm',
                        help=' trained model full path')

    args = parser.parse_args()

    # generate backend, it is optional to change to backend='mkl'
    be = gen_backend(backend='cpu', batch_size=10)

    # read training and validation data file
    with open(args.data_set_file, 'rb') as fp:
        data_in = pickle.load(fp)

    X_train = data_in['X_train']
    X_valid = data_in['X_valid']
    y_train = data_in['y_train']
    y_valid = data_in['y_valid']

    logger.info('training set size: ' + repr(len(y_train)))
    logger.info('validation set size: ' + repr(len(y_valid)))

    results = most_common_word_train(x_train=X_train, y_train=y_train, x_valid=X_valid, y_valid=y_valid)
