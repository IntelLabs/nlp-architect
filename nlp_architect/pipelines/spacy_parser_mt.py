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
from nlp_architect.common.core_nlp_doc import CoreNLPDoc
from nlp_architect.utils.text import SpacyInstance
from pathlib import Path
from joblib import Parallel, delayed
from functools import partial
from spacy.util import minibatch


class SpacyParserMT:
    """Main class which handles parsing with Spacy-BIST parser.

    Args:
        disable (list, optional):
        spacy_model (str, optional): Spacy model to use
        (see https://spacy.io/api/top-level#spacy.load).
    """
    def __init__(self, model='en_core_web_sm', disable=None):
        print("Loaded model '%s'" % model)
        self.nlp = SpacyInstance(model, disable=disable).parser

    def parse_multiple(self, doc_texts, output_dir, show_tok=True, show_doc=True, n_jobs=4, 
                        batch_size=1000):
        print("Processing texts...")
        partitions = minibatch(doc_texts, size=batch_size)
        print(batch_size)
        executor = Parallel(n_jobs=n_jobs, backend="multiprocessing", prefer="processes")
        do = delayed(partial(process_batch, self.nlp))
        tasks = (do(i, batch, output_dir, show_tok, show_doc) for i, batch in enumerate(partitions))
        executor(tasks)
        return 1000 # TODO: repleace with number of written jsons

def process_batch(nlp, batch_id, texts, output_dir, show_tok, show_doc):
    print("Processing batch", batch_id)
    for i, doc in enumerate(nlp.pipe(texts)):
        out_path = Path(output_dir) / ("{}.{}.json".format(batch_id, i))
        with out_path.open("w", encoding="utf8") as f:
            parsed_doc = CoreNLPDoc.from_spacy(doc, show_tok, show_doc, ptb_pos=True)
            f.write(parsed_doc.pretty_json())
    print("Saved {} CoreNLPDocs".format(len(texts)))