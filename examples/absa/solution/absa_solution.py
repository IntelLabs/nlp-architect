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
from os import path
from pathlib import Path

from nlp_architect.solutions.absa_solution.sentiment_solution import \
    SentimentSolution


def main():
    lib_root = Path(path.realpath(__file__)).parent.parent.parent.parent
    tripadvisor_data = lib_root / 'datasets' / 'absa' / \
        'tripadvisor_co_uk-travel_restaurant_reviews_sample_2000_test.csv'
    lexicons_dir = lib_root / 'examples' / 'absa' / 'inference'
    solution = SentimentSolution()
    solution.run(data=tripadvisor_data,
                 aspect_lex=lexicons_dir / 'aspects.csv',
                 opinion_lex=lexicons_dir / 'opinions.csv')


if __name__ == '__main__':
    main()
