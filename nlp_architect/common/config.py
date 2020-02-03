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
"""
Generic config object:
    load config from json file
    load config from ordinary python dict
    export config as dictionaty or json string
    define in init default parameters
"""

import copy
import json
import abc


class Config(abc.ABC):
    """Quantization Configuration Object"""

    ATTRIBUTES = {}

    def __init__(self, **kwargs):
        for entry in self.ATTRIBUTES:
            setattr(self, entry, kwargs.pop(entry, self.ATTRIBUTES[entry]))
        if kwargs:
            raise TypeError(f"got an unexpected keyword argument: {list(kwargs.keys())}")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a config from a Python dictionary of parameters."""
        config = cls()
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    def __repr__(self):
        return str(self.to_json_string())

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs Config from a json file of parameters."""
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))
