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

# This dataset should be downloaded from http://jmcauley.ucsd.edu/data/amazon/
# The terms and conditions of the data set license apply.
# Intel does not grant any rights to the data files.
# The Amazon Review Dataset was published in the following papers:
#
# Ups and downs:
# Modeling the visual evolution of fashion trends with one-class collaborative filtering
# R. He, J. McAuley
# WWW, 2016
# http://cseweb.ucsd.edu/~jmcauley/pdfs/www16a.pdf
#
# Image-based recommendations on styles and substitutes
# J. McAuley, C. Targett, J. Shi, A. van den Hengel
# SIGIR, 2015
# http://cseweb.ucsd.edu/~jmcauley/pdfs/sigir15.pdf

import pandas as pd
import json

from nlp_architect.utils.generic import normalize, balance


good_columns = [
    "overall",
    "reviewText",
    "summary"
]


def review_to_sentiment(review):
    # Review is coming in as overall (the rating, reviewText, and summary)
    # this then cleans the summary and review and gives it a positive or negative value
    norm_text = normalize(review[2] + " " + review[1])
    review_sent = ['neutral', norm_text]
    if review[0] > 3:
        review_sent = ['positive', norm_text]
    elif review[0] < 3:
        review_sent = ['negative', norm_text]

    return review_sent


class Amazon_Reviews(object):
    """
    Take the *.json file of Amazon reviews as downloaded from
    http://jmcauley.ucsd.edu/data/amazon/
    Then does data cleaning and balancing, as well as transforms the reviews 1-5 to a sentiment
    """
    def __init__(self, review_file, run_balance=True):
        self.run_balance = run_balance

        print("Parsing and processing json file")
        data = []

        with open(review_file, 'r') as f:
            for line in f:
                data_line = json.loads(line)
                selected_row = []
                for item in good_columns:
                    selected_row.append(data_line[item])
                # as we read in, clean
                data.append(review_to_sentiment(selected_row))

        # Not sure how to easily balance outside of pandas...but should replace eventually
        self.amazon = pd.DataFrame(data, columns=['Sentiment', 'clean_text'])
        self.all_text = self.amazon['clean_text']
        self.labels_0 = pd.get_dummies(self.amazon['Sentiment'])
        self.labels = self.labels_0.values
        self.text = self.amazon['clean_text'].values

    def process(self):
        self.amazon = self.amazon[self.amazon['Sentiment'].isin(['positive', 'negative'])]

        if self.run_balance:
            # balance it out
            self.amazon = balance(self.amazon)

        print("Sample Data")
        print(self.amazon[['Sentiment', 'clean_text']].head())

        # mapping of the labels with dummies (has headers)
        self.labels_0 = pd.get_dummies(self.amazon['Sentiment'])
        self.labels = self.labels_0.values
        self.text = self.amazon['clean_text'].values
