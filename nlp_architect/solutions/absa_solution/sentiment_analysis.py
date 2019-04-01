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
import json
from os import PathLike
from os.path import isdir
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from nlp_architect.common.core_nlp_doc import CoreNLPDoc
from nlp_architect.models.absa.inference.data_types import TermType, \
    SentimentDocEncoder, SentimentDoc
from nlp_architect.models.absa.inference.inference \
    import SentimentInference
from nlp_architect.models.absa.utils import load_opinion_lex
from nlp_architect.solutions.absa_solution import SENTIMENT_OUT
from nlp_architect.solutions.absa_solution.ui import serve_ui, _ui_format
from nlp_architect.solutions.absa_solution.utils import Anonymiser
from nlp_architect.utils.io import walk_directory, validate_existing_filepath, \
    validate_existing_directory


class SentimentSolution(object):
    """Main class for executing Sentiment Solution pipeline.

    Args:
        anonymiser (Anonymiser, optional): Method to anonymise events' text.
        max_events (int, optional): Maximum number of events to show for each aspect-polarity pair.
    """

    def __init__(self, anonymiser: Anonymiser = None, max_events: int = 400):
        self.anonymiser = anonymiser
        self.max_events = max_events
        SENTIMENT_OUT.mkdir(parents=True, exist_ok=True)

    def run(self, aspect_lex: PathLike = None, opinion_lex: PathLike = None,
            data: PathLike = None, parsed_data: PathLike = None,
            inference_results: PathLike = None, ui=True) -> Optional[pd.DataFrame]:

        opinions = load_opinion_lex(opinion_lex)
        if not opinions:
            raise ValueError('Empty opinion lexicon!')
        aspects = pd.read_csv(aspect_lex, header=None, encoding='utf-8')[0]
        if aspects.empty:
            raise ValueError('Empty aspect lexicon!')
        if inference_results:
            with open(inference_results) as f:
                results = json.loads(f.read(), object_hook=SentimentDoc.decoder)
        elif data or parsed_data:
            inference = SentimentInference(aspect_lex, opinions, parse=False)
            parse = None
            if not parsed_data:  # source data is raw text, need to parse
                from nlp_architect.pipelines.spacy_bist import SpacyBISTParser
                parse = SpacyBISTParser().parse

            results = {}
            print('Running inference on data files... (Iterating data files)')
            data_source = parsed_data if parsed_data else data
            for file, doc in self._iterate_docs(data_source):
                parsed_doc = parse(doc) if parse \
                    else json.loads(doc, object_hook=CoreNLPDoc.decoder)
                sentiment_doc = inference.run(parsed_doc=parsed_doc)
                if sentiment_doc:
                    results[file] = sentiment_doc
            with open(SENTIMENT_OUT / 'inference_results.json', 'w') as f:
                json.dump(results, f, cls=SentimentDocEncoder, indent=4, sort_keys=True)
        else:
            print('No input given. Please supply one of: '
                  'data directory, parsed data directory, or inference results.')
            return None

        print("\nComputing statistics...")
        stats = self._compute_stats(results, aspects, opinions)
        print("Done.")
        if ui:
            serve_ui(stats, aspects)
        return stats

    @staticmethod
    def _iterate_docs(data: PathLike) -> tuple:
        if isdir(data):
            for file, doc_text in tqdm(list(walk_directory(data))):
                yield file, doc_text
        else:
            with open(data, encoding='utf-8') as f:
                for i, doc_text in tqdm(enumerate(f), total=_line_count(data)):
                    yield str(i + 1), doc_text

    def _compute_stats(self, results: dict, aspects: list, opinion_lex: dict) -> pd.DataFrame:
        """Aggregates counts for each aspect-polarity pairs, with separate counts for in-domain
         only events.
        """
        index = pd.MultiIndex.from_product([aspects, ['POS', 'NEG'], [False, True]],
                                           names=['Aspect', 'Polarity', 'inDomain'])
        stats = pd.DataFrame(columns=['Quantity', 'Score'], index=index)
        stats[['Quantity', 'Score']] = stats[['Quantity', 'Score']].fillna(0)
        stats = stats.sort_index()
        scores = stats.copy()

        for doc in tqdm(results.values()):
            for sent in doc.sentences:
                for event in sent.events:
                    aspect = [t for t in event if t.type == TermType.ASPECT][0]
                    opinion = [t for t in event if t.type == TermType.OPINION][0]
                    score = aspect.score
                    key = aspect.text, aspect.polarity.name
                    count = self._add_event(stats, key, False, score)
                    in_domain = opinion_lex[opinion.text.lower()].is_acquired
                    count_dom = self._add_event(stats, key, True, score) if in_domain else -1

                    if count <= self.max_events:
                        sent_ui = _ui_format(sent, doc)
                        self._add_sentence(sent_ui, stats, scores, key, False, count, score)
                        if in_domain:
                            self._add_sentence(sent_ui, stats, scores, key, True, count_dom, score)
        for key in index:  # sort sentences according to their scores
            stats.loc[key, 2:] = stats.loc[key][2:][np.argsort(scores.loc[key][2:])].tolist()
        return stats

    def _add_sentence(self, sent_ui: str, stats: pd.DataFrame, scores: pd.DataFrame, key: tuple,
                      in_domain: bool, count: int, score: int) -> int:
        """Utility function for adding event sentence to output."""
        sent_ui = self.anonymiser.run(sent_ui) if self.anonymiser else sent_ui
        sent_key = key + (in_domain,), 'Sent_' + str(count)
        stats.at[sent_key] = sent_ui
        scores.at[sent_key] = -abs(score)
        return count

    @staticmethod
    def _add_event(df: pd.DataFrame, key: tuple, in_domain: bool, score: int) -> int:
        """Utility function for incrementing event counts."""
        key = key + (in_domain,)
        count = int(df.loc[key, 'Quantity']) + 1
        df.loc[key, 'Quantity'] = count
        df.loc[key, 'Score'] += score
        return count


def _line_count(file):
    """Utility function for getting number of lines in a text file."""
    with open(file) as f:
        return len(list(f))


def main() -> None:
    parser = argparse.ArgumentParser(description='Aspect-Based Sentiment Analysis')
    parser.add_argument('--data', type=validate_existing_directory,
                        help='Path to data')
    parser.add_argument('--aspects', type=validate_existing_filepath,
                        help='Path to aspect lexicon', required=True)
    parser.add_argument('--opinions', type=validate_existing_filepath,
                        help='Path to opinion lexicon', required=True)
    parser.add_argument('--parsed', type=validate_existing_directory,
                        help='Path to parsed data')
    parser.add_argument('--res', type=validate_existing_filepath,
                        help='Path to inference results')
    args = parser.parse_args()

    solution = SentimentSolution()
    solution.run(data=args.data,
                 parsed_data=args.parsed,
                 inference_results=args.res,
                 aspect_lex=args.aspects,
                 opinion_lex=args.opinions)


if __name__ == '__main__':
    main()
