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
from examples.word_language_model_with_tcn.adding_problem.adding_model import TCNForAdding
from examples.word_language_model_with_tcn.toy_data.adding import Adding


def test_tcn_adding():
    """
    Sanity check -
    Test function checks to make sure training loss drops to ~0 on small dummy dataset
    """
    n_features = 2
    hidden_sizes = [64] * 3
    kernel_size = 3
    dropout = 0.0
    seq_len = 10
    n_train = 5000
    n_val = 100
    batch_size = 32
    n_epochs = 10
    num_iterations = int(n_train * n_epochs * 1.0 / batch_size)
    lr = 0.002
    grad_clip_value = 10
    results_dir = "./"

    adding_dataset = Adding(seq_len=seq_len, n_train=n_train, n_test=n_val)

    model = TCNForAdding(seq_len, n_features, hidden_sizes, kernel_size=kernel_size,
                         dropout=dropout)

    model.build_train_graph(lr, max_gradient_norm=grad_clip_value)

    training_loss = model.run(adding_dataset, num_iterations=num_iterations, log_interval=1e6,
                              result_dir=results_dir)

    assert training_loss < 1e-3
