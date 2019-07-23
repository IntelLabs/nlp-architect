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
import os

import numpy as np
import pandas as pd
from pathlib import Path

from nlp_architect import LIBRARY_ROOT
from nlp_architect.solutions.absa_solution import SENTIMENT_OUT
from nlp_architect.solutions.absa_solution.sentiment_solution import \
    SentimentSolution
from nlp_architect.utils.io import download_unzip


def test_solution(generate_new=False):
    lexicons_dir = Path(LIBRARY_ROOT) / 'examples' / 'absa' / 'inference'
    expected_dir = Path(LIBRARY_ROOT) / 'tests' / 'fixtures' / 'data' / 'absa_solution'
    data_url = 'https://s3-us-west-2.amazonaws.com/nlp-architect-data/tests/'
    parsed_data = download_unzip(data_url, 'tripadvisor_test_parsed.zip',
                                 SENTIMENT_OUT / 'test' / 'tripadvisor_test_parsed')

    predicted_stats = SentimentSolution().run(parsed_data=parsed_data,
                                              aspect_lex=lexicons_dir / 'aspects.csv',
                                              opinion_lex=lexicons_dir / 'opinions.csv')

    predicted_stats.to_csv('predicted.csv', encoding='utf-8')
    predicted_trimmed = pd.read_csv('predicted.csv', encoding='utf-8').loc[:, 'Aspect': 'Score']
    predicted_trimmed.loc[:, 'Score'] = np.around(predicted_trimmed.loc[:, 'Score'], 2)
    os.remove('predicted.csv')

    if generate_new:
        with open('expected.csv', 'w', encoding='utf-8', newline='') as f:
            predicted_trimmed.to_csv(f)
        assert False

    else:
        with open(expected_dir / 'expected.csv', encoding='utf-8') as expected_fp:
            assert predicted_trimmed.to_csv() == expected_fp.read()
