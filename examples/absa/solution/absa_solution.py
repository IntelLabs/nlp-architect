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
from pathlib import Path

from nlp_architect import LIBRARY_ROOT, LIBRARY_DATASETS
from nlp_architect.solutions.absa_solution.sentiment_analysis import \
    SentimentSolution


def main():
    tripadvisor_data = LIBRARY_DATASETS / 'absa' / \
        'tripadvisor_co_uk-travel_restaurant_reviews_sample_2000_test.csv'
    lexicons_dir = Path(LIBRARY_ROOT) / 'examples' / 'absa' / 'inference'
    solution = SentimentSolution()
    solution.run(data=tripadvisor_data,
                 aspect_lex=lexicons_dir / 'aspects.csv',
                 opinion_lex=lexicons_dir / 'opinions.csv')


if __name__ == '__main__':
    main()
