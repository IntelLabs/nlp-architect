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
import json
from pathlib import Path

from nlp_architect import LIBRARY_ROOT
from nlp_architect.common.core_nlp_doc import CoreNLPDoc
from nlp_architect.models.absa.inference.data_types import SentimentDoc
from nlp_architect.models.absa.inference.inference import SentimentInference


def test_inference():
    lexicons_dir = Path(LIBRARY_ROOT) / "examples" / "absa"
    inference = SentimentInference(
        lexicons_dir / "aspects.csv", lexicons_dir / "opinions.csv", parse=False
    )
    data_dir = Path(LIBRARY_ROOT) / "tests" / "fixtures" / "data" / "absa"
    for i in range(1, 4):
        with open(data_dir / "core_nlp_doc_{}.json".format(i)) as f:
            predicted_doc = inference.run(parsed_doc=json.load(f, object_hook=CoreNLPDoc.decoder))
        with open(data_dir / "sentiment_doc_{}.json".format(i)) as f:
            expected_doc = json.load(f, object_hook=SentimentDoc.decoder)
        assert expected_doc == predicted_doc
