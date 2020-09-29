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
from pathlib import Path, PosixPath
from os import PathLike
from typing import Union

from nlp_architect.models.absa import TRAIN_OUT, LEXICONS_OUT
from nlp_architect.models.absa.train.acquire_terms import AcquireTerms
from nlp_architect.models.absa.train.rerank_terms import RerankTerms
from nlp_architect.models.absa.utils import (
    parse_docs,
    _download_pretrained_rerank_model,
    _write_aspect_lex,
    _write_opinion_lex,
)
from nlp_architect.utils.io import download_unzip

EMBEDDING_URL = "http://nlp.stanford.edu/data", "glove.840B.300d.zip"
EMBEDDING_PATH = TRAIN_OUT / "word_emb_unzipped" / "glove.840B.300d.txt"
RERANK_MODEL_DEFAULT_PATH = rerank_model_dir = TRAIN_OUT / "reranking_model" / "rerank_model.h5"


class TrainSentiment:
    def __init__(
        self,
        parse: bool = True,
        rerank_model: PathLike = None,
        asp_thresh: int = 3,
        op_thresh: int = 2,
        max_iter: int = 3,
    ):
        self.acquire_lexicon = AcquireTerms(asp_thresh, op_thresh, max_iter)
        if parse:
            from nlp_architect.pipelines.spacy_bist import SpacyBISTParser

            self.parser = SpacyBISTParser()
        else:
            self.parser = None

        if not rerank_model:
            print("using pre-trained reranking model")
            rerank_model = _download_pretrained_rerank_model(RERANK_MODEL_DEFAULT_PATH)

        download_unzip(*EMBEDDING_URL, EMBEDDING_PATH, license_msg="Glove word embeddings.")
        self.rerank = RerankTerms(
            vector_cache=True, rerank_model=rerank_model, emb_model_path=EMBEDDING_PATH
        )

    def run(
        self,
        data: Union[str, PathLike] = None,
        parsed_data: Union[str, PathLike] = None,
        out_dir: Union[str, PathLike] = TRAIN_OUT,
    ):

        if not parsed_data:
            if not self.parser:
                raise RuntimeError("Parser not initialized (try parse=True at init)")
            parsed_dir = Path(out_dir) / "parsed" / Path(data).stem
            parsed_data = self.parse_data(data, parsed_dir)

        generated_aspect_lex = self.acquire_lexicon.acquire_lexicons(parsed_data)
        _write_aspect_lex(parsed_data, generated_aspect_lex, LEXICONS_OUT)

        generated_opinion_lex_reranked = self.rerank.predict(
            AcquireTerms.acquired_opinion_terms_path, AcquireTerms.generic_opinion_lex_path
        )
        _write_opinion_lex(parsed_data, generated_opinion_lex_reranked, LEXICONS_OUT)

        return generated_opinion_lex_reranked, generated_aspect_lex

    def parse_data(self, data: PathLike or PosixPath, parsed_dir: PathLike or PosixPath):
        _, data_size = parse_docs(self.parser, data, out_dir=parsed_dir)
        if data_size < 1000:
            raise ValueError(
                "The data contains only {0} sentences. A minimum of 1000 "
                "sentences is required for training.".format(data_size)
            )
        return parsed_dir
