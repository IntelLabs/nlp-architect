# ******************************************************************************
# Copyright 2017-2019 Intel Corporation
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
# Register procedures to be used by CLI
from nlp_architect.cli.cmd_registry import CMD_REGISTRY
from nlp_architect.procedures.procedure import Procedure


def register_cmd(registry: dict, name: str, description: str):
    def register_cmd_fn(cls):
        if not issubclass(cls, Procedure):
            raise ValueError("Registered class must be subclassed from Procedure")
        if name in registry:
            raise ValueError("Cannot register duplicate model {}".format(name))
        run_fn = cls.run_procedure
        arg_adder_fn = cls.add_arguments
        new_cmd = {
            "name": name,
            "description": description,
            "fn": run_fn,
            "arg_adder": arg_adder_fn,
        }
        registry.append(new_cmd)
        return cls

    return register_cmd_fn


def register_train_cmd(name: str, description: str):
    return register_cmd(CMD_REGISTRY["train"], name, description)


def register_inference_cmd(name: str, description: str):
    return register_cmd(CMD_REGISTRY["inference"], name, description)
