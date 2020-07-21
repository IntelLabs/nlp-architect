# ******************************************************************************
# Copyright 2019-2020 Intel Corporation
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

import pytorch_lightning as pl
from bert_token_classifier import BertForToken, load_config, load_last_ckpt
from sys import argv

# pylint: disable=no-member
def main(config_yaml):
    config = load_config(config_yaml)
    pl.seed_everything(config.seed)

    model = BertForToken(config)
    trainer = model.get_trainer()

    if config.do_train:
        trainer.fit(model)

    trainer.logger.log_hyperparams(config)
    trainer.logger.save()

    if config.do_predict:        
        # Bug in pytorch_lightning==0.85 -> testing only works with num gpus=1
        trainer = model.get_trainer(gpus_override=1)
        trainer.test(load_last_ckpt(model, config))

if __name__ == "__main__":
    argv = ['', 'example']
    main(argv[1])
