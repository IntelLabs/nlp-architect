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
import csv
import logging
import operator
import os
import sys
from os import path
from pathlib import Path
from shutil import copyfile

import numpy
import pandas as pd
from fastText import load_model
from sklearn.manifold import TSNE

from nlp_architect import LIBRARY_OUT
from nlp_architect.utils.io import check_size, validate_existing_filepath
from nlp_architect.utils.text import simple_normalizer

dir = str(LIBRARY_OUT / 'trend-analysis-data')
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)
target_topics_path = path.join(dir, 'target_topics.csv')
ref_topics_path = path.join(dir, 'ref_topics.csv')


def analyze(target_data, ref_data, tar_header, ref_header, top_n=10000, top_n_vectors=500,
            re_analysis=False, tfidf_w=0.5, cval_w=0.5, lm_w=0):
    """
    Compare a topics list of a target data to a topics list of a reference data
    and extract hot topics, trends and clusters. Topic lists can be generated
    by running topic_extraction.py

    Args:
        target_data: A list of topics with importance scores extracted from the tagret corpus
        ref_data: A list of topics with importance scores extracted from the reference corpus
        tar_header: The header to appear for the target topics graphs
        ref_header: The header to appear for the reference topics graphs
        top_n (int): Limit the analysis to only the top N phrases of each list
        top_n_vectors (int): The number of vectors to include in the scatter
        re_analysis (Boolean): whether a first analysis has already been made or not
        tfidf_w (Float): the TF_IDF weight for the final score calculation
        cval_w (Float): the C_Value weight for the final score calculation
        lm_w (Float): the Language-Model weight for the final score calculation
    """
    hash2group = {}
    rep2rank = {}  # the merged list of groups extracted from both corpora.
    #  Will be sorted by their rank.
    in_model_count = 0
    create_clusters = False
    try:
        if not re_analysis:  # first analysis, not through ui
            copyfile(target_data, target_topics_path)
            copyfile(ref_data, ref_topics_path)
        calc_scores(target_data, tfidf_w, cval_w, lm_w, target_topics_path)
        calc_scores(ref_data, tfidf_w, cval_w, lm_w, ref_topics_path)
        # unify all topics:
        with open(ref_data) as f:
            topics1 = sum(1 for _ in f)
        with open(target_data) as f:
            topics2 = sum(1 for _ in f)
        sum_topics = topics1 + topics2
        logger.info("sum of all topics= %s", str(sum_topics))
        merge_phrases(ref_topics_path, True, hash2group, rep2rank, top_n, sum_topics)
        merge_phrases(target_topics_path, False, hash2group, rep2rank, top_n, sum_topics)
        logger.info("Total number of evaluated topics: %s", str(len(rep2rank)))

        # compute 2D space clusters if model exists:
        w2v_loc = path.join(dir, 'W2V_Models/model.bin')
        if os.path.isfile(w2v_loc):
            all_topics_sorted = sorted(rep2rank, key=rep2rank.get)
            top_n_scatter = len(
                all_topics_sorted) if top_n_vectors is None else top_n_vectors
            scatter_group = all_topics_sorted[0:top_n_scatter]
            np_scat, x_scat, y_scat, in_model_count =\
                compute_scatter_subwords(scatter_group, w2v_loc)
            if np_scat is not None and x_scat is not None:
                create_clusters = True
                for j in range(len(np_scat)):
                    hash2group[simple_normalizer(np_scat[j])] += (x_scat[j],
                                                                  y_scat[j])
        # prepare reports data:
        groups_r = list(
            filter(lambda x: x[2] == 0 or x[2] == 2, hash2group.values()))
        groups_t = list(
            filter(lambda x: x[2] == 1 or x[2] == 2, hash2group.values()))
        trends = list(
            filter(lambda x: x[2] == 2, hash2group.values()))  # all trends
        groups_r_sorted = sorted(
            groups_r, key=operator.itemgetter(1), reverse=True)
        groups_t_sorted = sorted(
            groups_t, key=operator.itemgetter(3), reverse=True)
        trends_sorted = sorted(
            trends, key=operator.itemgetter(6), reverse=True)  # sort by t_score

        # save results:
        save_report_data(hash2group, groups_r_sorted, groups_t_sorted,
                         trends_sorted, all_topics_sorted, create_clusters,
                         tar_header, ref_header, tfidf_w, cval_w, lm_w,
                         in_model_count, top_n_scatter)
        logger.info('Done analysis.')
    except Exception as e:
        logger.error(str(e))


def save_report_data(hash2group, groups_r_sorted, groups_t_sorted,
                     trends_sorted, all_topics_sorted, create_clusters,
                     target_header, ref_header, tfidf_w, cval_w, freq_w,
                     in_model_count, top_n_scatter):
    filter_output = path.join(dir, 'filter_phrases.csv')
    data_output = path.join(dir, 'graph_data.csv')
    logger.info('writing results to: %s and: %s ', str(data_output), str(filter_output))

    reports_column = [
        'Top Topics (' + ref_header + ')',
        'Top Topics (' + target_header + ')',
        'Hot Trends',
        'Cold Trends',
        'Custom Trends'
    ]
    if create_clusters:
        reports_column.extend([
            'Trend Clustering',
            'Topic Clustering (' + ref_header + ')',
            'Topic Clustering (' + target_header + ')'
        ])
    headers = [
        'reports',
        'ref_topic', 'ref_imp', 'x_ref', 'y_ref',
        'tar_topic', 'tar_imp', 'x_tar', 'y_tar',
        'trends', 'change', 't_score', 'x_tre', 'y_tre', 'weights',
        'in_w2v_model_count', 'top_n_scatter'
    ]
    filter_headers = [
        'topics',
        'valid',
        'custom'
    ]
    weights = [tfidf_w, cval_w, freq_w]

    with open(data_output, 'wt', encoding='utf-8') as data_file:
        with open(filter_output, 'wt', encoding='utf-8') as filter_file:
            data_writer = csv.writer(data_file, delimiter=',')
            filter_writer = csv.writer(filter_file, delimiter=',')
            data_writer.writerow(headers)
            filter_writer.writerow(filter_headers)
            for i in range(len(hash2group.keys())):
                try:
                    new_row = ()
                    new_row += (reports_column[i],) if i < len(
                        reports_column) else ('',)
                    if i < len(groups_r_sorted):
                        new_row += (
                            groups_r_sorted[i][0], groups_r_sorted[i][1])
                        # 'only b' type tuple
                        if groups_r_sorted[i][2] == 0:
                            new_row += (
                                groups_r_sorted[i][-2],
                                groups_r_sorted[i][-1]) if len(
                                groups_r_sorted[i]) > 4 else (-1, -1)
                        else:  # 'trend' type tuple
                            new_row += (
                                groups_r_sorted[i][-2],
                                groups_r_sorted[i][-1]) if len(
                                groups_r_sorted[i]) > 7 else (-1, -1)
                    else:
                        new_row += ('', '', '', '')
                    if len(groups_t_sorted) > i:
                        new_row += (
                            groups_t_sorted[i][0], groups_t_sorted[i][3])
                        # 'only a' type tuple
                        if groups_t_sorted[i][2] == 1:
                            new_row += (groups_t_sorted[i][-2],
                                        groups_t_sorted[i][-1]) if len(
                                groups_t_sorted[i]) > 5 else (-1, -1)
                        else:
                            new_row += (groups_t_sorted[i][-2],
                                        groups_t_sorted[i][-1]) if len(
                                groups_t_sorted[i]) > 7 else (-1, -1)
                    else:
                        new_row += ('', '', '', '')
                    if len(trends_sorted) > i:
                        new_row += (
                            trends_sorted[i][0], trends_sorted[i][4], trends_sorted[i][6])
                        new_row += (
                            trends_sorted[i][-2],
                            trends_sorted[i][-1]) if len(
                            trends_sorted[i]) > 7 else (-1, -1)
                    else:
                        new_row += ('', '', '', '')
                    new_row += (weights[i],) if i < len(
                        weights) else ('',)
                    new_row += (in_model_count, top_n_scatter) if i == 0 else ('',)
                    data_writer.writerow(new_row)

                    filter_row = (all_topics_sorted[i], 1, 0)
                    filter_writer.writerow(filter_row)
                except Exception as e:
                    logger.error(
                        "Error while writing analysis results. "
                        "iteration #: %s. Error: %s", str(i), str(e))
            if len(hash2group.keys()) < len(reports_column):
                for i in range(len(hash2group.keys()), len(reports_column)):
                    new_row = ()
                    new_row += (reports_column[i], '', '', '', '', '', '', '',
                                '', '', '', '', '', '', '', '', '')
                    data_writer.writerow(new_row)


def compute_scatter_subwords(top_groups, w2v_loc):
    """
    Compute 2D vectors of the provided phrases groups

    Args:
        top_groups: A list of group-representative phrases (List(String))
        w2v_loc: A path to a w2v model (String)

    Returns:
        A tuple (phrases, x, y, n)
        WHERE:
        phrases: A list of phrases that are part of the model
        x: A DataFrame column as the x values of the phrase vector
        y: A DataFrame column as the y values of the phrase vector
        n: The number of computed vectors
    """
    x_feature, np_indices = [], []
    in_ctr = 0

    try:
        if path.isfile(w2v_loc):
            logger.info("computing scatter on %s groups", str(len(top_groups)))
            model = load_model(w2v_loc)
            for a in top_groups:
                x_feature.append(numpy.array(model.get_word_vector(a)))
                np_indices.append(a)
                in_ctr += 1
            logger.info("topics found in model out of %s: %s.", str(len(top_groups)), str(in_ctr))
            if len(np_indices) == 0:
                logger.error('no vectors extracted')
                return None, None, None, 0
            logger.info('computing TSNE embedding...')
            tsne = TSNE(n_components=2, random_state=0, method='exact')
            numpy.set_printoptions(suppress=True)
            x_tsne = tsne.fit_transform(x_feature)
            df = pd.DataFrame(
                x_tsne, index=np_indices, columns=['x', 'y'])
            return np_indices, df['x'], df['y'], in_ctr
        logger.error('no model found for cumputing scatter, skipping step.')
        return None, None, None, 0
    except Exception as e:
        logger.error(str(e))


def calc_scores(scores_file, tfidf_w, cval_w, lm_w, output_path):
    """
    Given a topic list with tf_idf,c_value,language_model scores, compute
     a final score for each phrases group according to the given weights

    Args:
        scores_file (String): A path to the file with groups and raw scores
        tfidf_w (Float): the TF_IDF weight for the final score calculation
        cval_w (Float): the C_Value weight for the final score calculation
        lm_w (Float): the Language-Model weight for the final score calculation
        output_path: A path for the output file of final scores (String)
    """
    logger.info("calculating scores for file: %s with: tfidf_w=%s, cval_w=%s,"
                " freq_w=%s", scores_file, str(tfidf_w), str(cval_w), str(lm_w))
    with open(scores_file, encoding='utf-8', errors='ignore') as csv_file:
        lines = csv.reader(csv_file, delimiter=',')
        final_list = []
        for group, tfidf, cval, freq in lines:
            score = (float(tfidf) * tfidf_w) + (float(cval) * cval_w) + (float(freq) * lm_w)
            final_list.append((group, score))
        sorted_list = sorted(final_list, key=lambda tup: tup[1], reverse=True)
    save_scores_list(sorted_list, output_path)


def save_scores_list(scores, file_path):
    """
    Save the list of topics-scores into a file

    Args:
        scores: A list of topics (groups) with scores
        file_path: The output file path
    """
    logger.info('saving np extraction results to: %s', file_path)
    with open(file_path, 'wt', encoding='utf-8', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        ctr = 0
        for topics, imp in scores:
            try:
                row = (topics, str(imp))
                writer.writerow(row)
                ctr += 1
            except Exception as e:
                logger.error("Error while writing scores to file. iteration #: %s. Error: %s", str(
                    ctr), str(e))


def merge_phrases(data, is_ref_data, hash2group, rep2rank, top_n, topics_count):
    """
    Analyze the provided topics data and detect trends (changes in importance)

    Args:
        data: A list of topics with importance scores
        is_ref_data (Boolean): Was the data extracted from the target/reference corpus
        hash2group: A dictionary storing the data of each topic
        rep2rank: A dict of all groups representatives and their ranks
        top_n (int): Limit the analysis to only the top N phrases of each list
        topics_count (int): The total sum of all topics extracted from both corpora
    """
    logger.info('merge and compare groups for data: %s', str(data))
    ctr = 0
    if not Path(data).exists():
        logger.error('invalid csv file: %s', str(data))
        sys.exit()
    try:
        with open(data, encoding='utf-8', errors='ignore') as csv_file:
            topics = csv.reader(csv_file, delimiter=',')
            for group, imp in topics:
                if ctr == top_n:
                    break
                try:
                    rep = clean_group(group).strip()
                    imp = float(imp) * 100.0
                    rank = ctr + 1
                    hash_id = simple_normalizer(rep)
                    if hash_id not in hash2group:
                        rep2rank[rep] = rank
                        if is_ref_data:
                            hash2group[hash_id] = (rep, imp, 0, rank)
                        else:
                            hash2group[hash_id] = (rep, 0, 1, imp, rank)
                    elif not is_ref_data:
                        data_b = hash2group[hash_id]
                        if data_b[2] == 0:  # create a trend only in comparison to the
                            #  ref topics, ignore cases of different topics have the
                            # same hash or same topic was extracted twice from the
                            # same data
                            old_rep = data_b[0]
                            old_rank = data_b[3]
                            rep2rank[old_rep] = int((rank + old_rank) / 2)  # rank
                            #  of topic that appear in both corpora calculated as
                            #  the avarage of ranks
                            change = float(imp) - float(data_b[1])
                            t_score = (topics_count - (old_rank + rank)) * abs(change)
                            hash2group[hash_id] = (
                                old_rep, float(data_b[1]), 2, imp, change,
                                abs(change), t_score)  # trend phrase
                    ctr += 1
                except Exception as e:
                    logger.error('bad line: %s. Error: %s', str(ctr), str(e))
    except Exception as e:
        logger.error('Error: %s. Is %s a valid csv file?', str(e), str(data))
        sys.exit()


def clean_group(phrase_group):
    """
   Returns the shortest element in a group of phrases

    Args:
        phrase_group (String): a group of phrases separated by ';'

    Returns:
        The shortest phrase in the group (String)
    """
    text = [x.lstrip() for x in phrase_group.split(';')]
    return min(text, key=len)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='trend_analysis.py')
    parser.add_argument('target_topics', metavar='target_topics', type=validate_existing_filepath,
                        help='a path to a csv topic-list extracted from the '
                             'target corpus')
    parser.add_argument('ref_topics', metavar='ref_topics', type=validate_existing_filepath,
                        help='a path to a csv topic-list extracted from the '
                             'reference corpus')
    parser.add_argument('--top_n', type=int, action=check_size(0, 100000), default=10000,
                        help='compare only top N topics (default: 10000)')
    parser.add_argument('--top_vectors', type=int, action=check_size(0, 100000), default=500,
                        help='include only top N vectors in the scatter graph (default: 500)')
    args = parser.parse_args()
    analyze(args.target_topics, args.ref_topics, args.target_topics,
            args.ref_topics, args.top_n, args.top_vectors)
