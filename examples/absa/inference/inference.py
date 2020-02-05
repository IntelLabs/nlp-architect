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

from nlp_architect.models.absa.inference.inference import SentimentInference


def main() -> list:
    parent_dir = Path(path.realpath(__file__)).parent.parent
    inference = SentimentInference(parent_dir / "aspects.csv", parent_dir / "opinions.csv")

    print("\n" + "=" * 40 + "\n" + "Running inference on examples from sample test set:\n")

    docs = [
        "The food was very fresh and flavoursome the service was very attentive. Would go back"
        " to this restaurant if visiting London again.",
        "The food was wonderful and fresh, I really enjoyed it and will definitely go back. "
        "Staff were friendly.",
        "The ambiance is charming. Uncharacteristically, the service was DREADFUL. When we"
        " wanted to pay our bill at the end of the evening, our waitress was nowhere to be "
        "found...",
    ]

    sentiment_docs = []

    for doc_raw in docs:
        print("Raw Document: \n{}".format(doc_raw))
        sentiment_doc = inference.run(doc=doc_raw)
        sentiment_docs.append(sentiment_doc)
        print("SentimentDocument: \n{}\n".format(sentiment_doc) + "=" * 40 + "\n")
    return sentiment_docs


if __name__ == "__main__":
    main()
