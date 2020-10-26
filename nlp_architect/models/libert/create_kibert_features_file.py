# ******************************************************************************
# Copyright 2019-2020 Intel Corporation
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
"""Helper script to create cached features file for kibert"""

import os
import argparse
from pathlib import Path
from os.path import realpath
import torch
import datasets
from transformers import (
    BertTokenizer,
    DPRQuestionEncoderTokenizer,
    DPRQuestionEncoder,
)
from tqdm import tqdm
import absa_utils

LIBERT_DIR = Path(realpath(__file__)).parent

def main(args):
    if args.data_dir:
        data_dir = args.data_dir
    else:
        print("Need to supply a path to data_dir.")
        print("Example usage: 'python create_kibert_features_file.py --data_dir /PATH_TO_DATA/laptops_to_restaurants_1'")
        exit()

    if os.path.basename(data_dir) == '':
        print("Please make sure that there is no '/' at the end of the data dir path, e.g. it should be path/laptops_to_restaurants_1 instead of path/laptops_to_restaurants_1/. Exiting.")
        exit()
    
    if args.merge_train_dev:
        merge_train_dev = True
    else:
        merge_train_dev = False

    if args.query_domain:
        query_domain = args.query_domain
    else:
        print("No query domain specified, defaulting to 'test'.")
        query_domain = "test"

    data_root = Path(data_dir).parent
    labels = absa_utils.get_labels(data_root / "labels.txt")
    max_seq_length = 64
    cache_dir = LIBERT_DIR / 'cache'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=cache_dir)

    # Notice messages
    print(f"\nCreating cached features files in {data_dir}.")
    print(f"Please make sure that libert/data/csv points to {data_root} for your model to use these cached features file.")
    print(f"e.g. if you are in the libert directory, you can use the following command 'ln -s {data_root} ./data/csv' to link to the newly created cached features file.")
    print(f"Also note that you should set the 'overwrite_cache' option in your config file to 'false', and you will have to set the 'data' and 'splits' options according to the given data directory, e.g. laptops_to_restaurants_1 will correspond to 'lr' and '1' for 'data' and 'splits' respectively. \n")

    print("Loading DPR and wiki_dpr dataset...")
    rag_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    rag_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    wiki_dataset =  datasets.load_dataset('wiki_dpr')

    for mode in "train", "dev", "test":
        print(f"Creating cached features file for {mode}...")
        cached_features_file = os.path.join(data_dir, f"cached_{mode}_bert-base-uncased_64")
        examples = absa_utils.read_examples_from_file(data_dir, mode)
        features = absa_utils.convert_examples_to_features(
            examples,
            labels,
            max_seq_length,
            tokenizer
        )

        # Ex: dataset_name = "laptops_to_restaurants_1"
        dataset_name = os.path.basename(data_dir)
        train_domain = dataset_name.split('_')[0]
        val_test_domain = dataset_name.split('_')[2]

        print(f"Constructing DPR features based on the following query: how is [aspect] related to [{query_domain} domain]")
        if query_domain == "respective":
            domain = train_domain if mode=="train" else val_test_domain
        elif query_domain == "test":
            domain = val_test_domain
        elif query_domain == "train":
            domain = train_domain
        else:
            print(f"Unrecognized query domain '{query_domain}'")
        
        new_features = []
        k = 5  # number of embeddings to retrieve
        for e, f in tqdm(zip(examples, features), total=len(examples)):
            # look for aspect, TODO: make this noun instead?
            # TODO: decide on what to do if no B-ASP?
            #try:
            asp_idx = e.labels.index('B-ASP')
            asp = e.words[asp_idx]
            question = f"how is {asp} related to {domain}"
            question_emb = rag_encoder(**rag_tokenizer(question, return_tensors="pt"))[0].detach().numpy()
            passages_scores, passages = wiki_dataset['train'].get_nearest_examples("embeddings", question_emb, k=k)  # get k nearest
            passage_embeddings = torch.tensor(passages['embeddings'])
            new_features.append((f, passage_embeddings))

            #except ValueError:
                # In case no B-ASP in example
            #    passage_embeddings = torch.zeros(5, 768)
            #    new_features.append((f, passage_embeddings)) 

        # Merging dev data into train data
        if mode == "train" and merge_train_dev == True:
            print("Adding dev data to train data...")
            examples_dev = absa_utils.read_examples_from_file(data_dir, "dev")
            features_dev = absa_utils.convert_examples_to_features(
                examples_dev,
                labels,
                max_seq_length,
                tokenizer
            )

            for e, f in tqdm(zip(examples_dev, features_dev), total=len(examples_dev)):
                # look for aspect, TODO: make this noun instead?
                # TODO: decide on what to do if no B-ASP?
                #try:
                asp_idx = e.labels.index('B-ASP')
                asp = e.words[asp_idx]
                question = f"how is {asp} related to {val_test_domain}"
                question_emb = rag_encoder(**rag_tokenizer(question, return_tensors="pt"))[0].detach().numpy()
                passages_scores, passages = wiki_dataset['train'].get_nearest_examples("embeddings", question_emb, k=k)  # get k nearest
                passage_embeddings = torch.tensor(passages['embeddings'])
                new_features.append((f, passage_embeddings))

                #except ValueError:
                    # In case no B-ASP in example
                #    passage_embeddings = torch.zeros(5, 768)
                #    new_features.append((f, passage_embeddings)) 

        torch.save(new_features, cached_features_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Path to data directory, e.g. /PATH_TO_DATA/laptops_to_restaurants_1")
    parser.add_argument("--merge_train_dev", action='store_true', help="Set if you want to merge the train and dev sets.")
    parser.add_argument("--query_domain", help="Choose which domain to query against, options are 'train', 'test' and 'respective'")
    args = parser.parse_args()
    main(args)
