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
import tempfile

from tensorflow import keras


def save_model(model: keras.models.Model, topology: dict, filepath: str) -> None:
    """
    Save a model to a file (tf.keras models only)
    The method save the model topology, as given as a
    Args:
        model: model object
        topology (dict): a dictionary of topology elements and their values
        filepath (str): path to save model
    """
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=True) as fd:
        model.save_weights(fd.name)
        model_weights = fd.read()
    data = {'model_weights': model_weights,
            'model_topology': topology}
    with open(filepath, 'wb') as fp:
        pickle.dump(data, fp)


def load_model(filepath, model) -> None:
    """
    Load a model (tf.keras) from disk, create topology from loaded values
    and load weights.
    Args:
        filepath (str): path to model
        model: model object to load
    """
    with open(filepath, 'rb') as fp:
        model_data = pickle.load(fp)
    topology = model_data['model_topology']
    model.build(**topology)
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=True) as fd:
        fd.write(model_data['model_weights'])
        fd.flush()
        model.model.load_weights(fd.name)
