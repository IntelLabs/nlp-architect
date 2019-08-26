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
import argparse
from os import path
from pathlib import Path

from nlp_architect.models.absa.train.train import TrainSentiment
from nlp_architect.utils.io import validate_existing_filepath, validate_existing_path, \
    validate_existing_directory


def main() -> None:
    lib_root = Path(path.realpath(__file__)).parent.parent.parent.parent
    tripadvisor_train = lib_root / 'datasets' / 'absa' / \
        'tripadvisor_co_uk-travel_restaurant_reviews_sample_2000_train.csv'

    parser = argparse.ArgumentParser(description='ABSA Train')
    parser.add_argument('--rerank-model', type=validate_existing_filepath,
                        default=None, help='Path to rerank model .h5 file')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--data', type=validate_existing_path,
                       default=tripadvisor_train,
                       help='Path to raw data (directory or txt/csv file)')
    group.add_argument('--parsed-data', type=validate_existing_directory, default=None,
                       help='Path to parsed data directory')
    args = parser.parse_args()

    train = TrainSentiment(parse=not args.parsed_data, rerank_model=args.rerank_model)
    opinion_lex, aspect_lex = train.run(data=args.data, parsed_data=args.parsed_data)

    print('Aspect Lexicon: {}\n'.format(aspect_lex) + '=' * 40 + '\n')
    print('Opinion Lexicon: {}'.format(opinion_lex))


if __name__ == '__main__':
    main()
