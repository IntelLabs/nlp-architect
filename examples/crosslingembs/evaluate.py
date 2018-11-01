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
from __future__ import print_function, division

import numpy as np
import tensorflow as tf

from nlp_architect.data.fasttext_emb import get_eval_data


class Evaluate:
    """
    Class for evaluating the performance of mapping W
    """

    def __init__(self, W, src_vec, tgt_vec, src_dico, tgt_dico, src_lang,
                 tgt_lang, eval_path, vocab_size):
        self.W = W
        self.src_ten = src_vec
        self.tgt_ten = tgt_vec
        self.src_dico = src_dico
        self.tgt_dico = tgt_dico
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.eval_path = eval_path
        self.vocab_size = vocab_size
        self.best_cos_score = -1e9
        self.drop_lr = False
        self.second_drop = False
        self.save_model = False
        self.max_dict_size = 15000

        # Lookup ids
        self.tgt_ids = [x for x in range(self.vocab_size)]

        # Placeholders
        self.src_ph = tf.placeholder(tf.int32, shape=[None], name="EvalSrcPh")
        self.tgt_ph = tf.placeholder(tf.int32, shape=[None], name="EvalTgtPh")
        self.score_ph = tf.placeholder(tf.float32, shape=[None, self.vocab_size], name="ScorePh")

        # Preprocessing for evaluation
        self.load_eval_dataset()
        self.prepare_emb()

        # Graph for evaluation
        self.eval_nn = self.build_eval_graph("NNEval", self.src_ten, self.tgt_ten, en_mean=False)
        self.csls_subgraphs = self.build_csls_subgraphs()

    def load_eval_dataset(self):
        """
        This method read through the test file and gets their embedding indices
        """
        dict_path = get_eval_data(self.eval_path, self.src_lang, self.tgt_lang)

        pairs = []
        not_found_all = 0
        not_found_L1 = 0
        not_found_L2 = 0

        # Open the file and check if src and tgt word exists in the vocab
        with open(dict_path, 'r') as f:
            for _, line in enumerate(f):
                word1, word2 = line.rstrip().split()
                if word1 in self.src_dico and word2 in self.tgt_dico:
                    pairs.append((self.src_dico.index(word1), self.tgt_dico.index(word2)))
                else:
                    not_found_all += 1
                    not_found_L1 += int(word1 not in self.src_dico)
                    not_found_L2 += int(word2 not in self.tgt_dico)
        print("Found %i pairs of words in the dictionary (%i unique). "
              " %i other pairs contained at least one unknown word "
              " (%i in src_lang, %i in tgt_lang)"
              % (len(pairs), len(set([x for x, _ in pairs])), not_found_all,
                 not_found_L1, not_found_L2))
        src_ind = [pairs[x][0] for x in range(len(pairs))]
        tgt_ind = [pairs[x][1] for x in range(len(pairs))]
        self.src_ind = np.asarray(src_ind)
        self.tgt_ind = np.asarray(tgt_ind)

    def prepare_emb(self):
        """
        Maps source embedding with the help of learnt mapping W.
        It also normalizes the embeddings to make it easy for similarity measurements.
        """
        with tf.variable_scope("PrepEmb", reuse=tf.AUTO_REUSE):
            self.src_ten = tf.cast(tf.convert_to_tensor(self.src_ten), tf.float32)
            self.tgt_ten = tf.cast(tf.convert_to_tensor(self.tgt_ten), tf.float32)
            # Mapping
            self.src_ten = tf.matmul(self.src_ten, self.W)
            # Normalization
            self.src_ten = tf.nn.l2_normalize(self.src_ten, axis=1)
            self.tgt_ten = tf.nn.l2_normalize(self.tgt_ten, axis=1)

    def build_eval_graph(self, name, emb_ten, query_ten, knn=100, en_knn=True, en_mean=True):
        """
        Build a simple evaluation graph

        Arguments:
            name (str) : Name of the graph
            emb_ten (tensor): Embedding tensor
            query_ten (tensor): Query tensor
            knn (int): K value for NN
            en_knn (bool): enable or disable knn portion of the graph
            en_mean (bool): enable or disabel mean portion of the graph

        Returns:
            Returns a graph depending on booleans above
        """
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # Look Up the words
            emb_lut = tf.nn.embedding_lookup(emb_ten, self.src_ph, name="EvalSrcLut")
            query_lut = tf.nn.embedding_lookup(query_ten, self.tgt_ph, name="EvalTgtLut")
            # Cast
            emb = tf.cast(emb_lut, tf.float32)
            query = tf.cast(query_lut, tf.float32)
            # MM
            sim_scores = tf.matmul(emb, tf.transpose(query))
            # Topk
            if en_knn and not en_mean:
                top_matches = tf.nn.top_k(sim_scores, knn)
                return top_matches
            if en_knn and en_mean:
                top_matches = tf.nn.top_k(sim_scores, knn)
                best_distances = tf.reduce_mean(top_matches[0], axis=1)
                return best_distances
            return sim_scores

    def build_sim_graph(self, name):
        """
        Builds a similarity measurement graph

        Arguments:
            name(str): Name of the graph
        Returns:
            Returns a handle to the graph
        """
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # Look up the words
            emb_lut = tf.nn.embedding_lookup(self.src_ten, self.src_ph, name="Src_Lut")
            query_lut = tf.nn.embedding_lookup(self.tgt_ten, self.tgt_ph, name="Tgt_Lut")
            # Cast
            emb = tf.cast(emb_lut, tf.float32)
            query = tf.cast(query_lut, tf.float32)
            # Score it
            score = emb * query
            cos_sum = tf.reduce_sum(score, axis=1)
            cos_mean = tf.reduce_mean(cos_sum)
            return cos_mean

    def build_csls_subgraphs(self):
        """
        Builds various graphs needed for CSLS calculation

        Returns:
            Returns a dictionary with various graphs constructed

        """
        # Graph for calculating only score
        scores_s2t = self.build_eval_graph(
            "ScoreS2T",
            self.src_ten,
            self.tgt_ten,
            en_knn=False,
            en_mean=False)
        scores_t2s = self.build_eval_graph(
            "ScoreT2S",
            self.tgt_ten,
            self.src_ten,
            en_knn=False,
            en_mean=False)
        # Graphs for calculating average between source and target
        avg1_s2t = self.build_eval_graph("Avg1", self.src_ten, self.tgt_ten, knn=10)
        avg2_s2t = self.build_eval_graph("Avg2", self.tgt_ten, self.src_ten, knn=10)
        # Graph for selecting top 100 elements
        top100_matches = tf.nn.top_k(self.score_ph, 100)
        # Graph for selecting top 2 elements
        top2_matches = tf.nn.top_k(self.score_ph, 2)
        # Graph for calculating similarity
        csls_mean_score = self.build_sim_graph("SimGraph")

        # Dictionary
        csls_graphs = {"ScoreGraph": scores_s2t,
                       "ScoreG_T2S": scores_t2s,
                       "Avg1S2T": avg1_s2t,
                       "Avg2S2T": avg2_s2t,
                       "Top100": top100_matches,
                       "Top2": top2_matches,
                       "CSLS_Criteria": csls_mean_score
                       }
        return csls_graphs

    def calc_nn_acc(self, sess, batch_size=512):
        """
        Evaluates accuracy of mapping using Nearest neighbors
        Arguments:
            sess(tf.session): Tensorflow Session
            batch_size(int): Size of batch
        """
        top_matches = []
        eval_size = len(self.src_ind)

        # Loop through all the eval dataset
        for i in range(0, eval_size, batch_size):
            src_ids = [self.src_ind[x] for x in range(i, min(i + batch_size, eval_size))]
            eval_dict = {self.src_ph: src_ids,
                         self.tgt_ph: self.tgt_ids}
            matches = sess.run(self.eval_nn, feed_dict=eval_dict)
            top_matches.append(matches[1])
        top_matches = np.concatenate(top_matches)

        print("Accuracy using Nearest Neighbors is")
        self.calc_accuracy(top_matches)

    def calc_accuracy(self, top_matches):
        """
        Takes top_matches generated by tf.nn.top_k and calculates accuracy
        Arguments:
             top_matches: Output of tf.nn.topk_k[1]
        """
        # Calculate Translation accuracy
        # Accuracy at K
        for k in [1, 5, 10]:
            top_k_matches = top_matches[:, :k]
            # Checks for one match
            one_matching = (top_k_matches == self.tgt_ind[:, None]).sum(1)
            # Checks for multiple translations
            matching = {}
            for i, src_id in enumerate(self.src_ind):
                matching[src_id] = min(matching.get(src_id, 0) + one_matching[i], 1)
            precision_at_k = 100 * np.mean(list(matching.values()))
            print("%i source words  - Precision at k = %i: %f" %
                  (len(matching), k, precision_at_k))

    def calc_csls_score(self, sess, batch_size=512):
        """
        Calculates similarity score between two embeddings
        Arguments:
            sess(tf.session): Tensorflow Session
            batch_size(int): Size of batch to process
        Returns:
            Returns similarity score numpy array
        """
        score_val = []
        eval_size = len(self.src_ind)
        # Calculate scores
        for i in range(0, eval_size, batch_size):
            score_src_ids = [self.src_ind[x] for x in range(i, min(i + batch_size, eval_size))]
            eval_dict = {self.src_ph: score_src_ids,
                         self.tgt_ph: self.tgt_ids}
            score_val.append(sess.run(self.csls_subgraphs["ScoreGraph"], feed_dict=eval_dict))
        score_val = np.concatenate(score_val)
        return score_val

    def calc_avg_dist(self, sess, batch_size=512):
        """
        Calculates average distance between two embeddings
        Arguments:
            sess(tf.session): Tensorflow session
            batch_size(int): batch_size
        Returns:
            Returns numpy array of average values of size vocab_size
        """
        avg1_val = []
        avg2_val = []

        # Calculate Average
        for i in range(0, self.vocab_size, batch_size):
            avg_src_ids = [x for x in range(i, min(i + batch_size, self.vocab_size))]
            avg1_dict = {self.src_ph: avg_src_ids,
                         self.tgt_ph: self.tgt_ids}
            avg1_val.append(sess.run(self.csls_subgraphs["Avg1S2T"], feed_dict=avg1_dict))
            avg2_val.append(sess.run(self.csls_subgraphs["Avg2S2T"], feed_dict=avg1_dict))
        avg1_val = np.concatenate(avg1_val)
        avg2_val = np.concatenate(avg2_val)
        return avg1_val, avg2_val

    def run_csls_metrics(self, sess, batch_size=512):
        """
        Runs the whole CSLS metrics
        Arguments:
            sess(tf.session): Tensorflow Session
            batch_size(int): Batch Size
        """
        top_matches = []
        score = self.calc_csls_score(sess)
        avg1, avg2 = self.calc_avg_dist(sess)
        csls_scores = 2 * score - (avg1[self.src_ind][:, None] + avg2[None, :])
        # Calculate top matches
        for i in range(0, len(self.src_ind), batch_size):
            scores = [csls_scores[x] for x in range(i, min(i + batch_size, len(self.src_ind)))]
            top_matches_val = sess.run(self.csls_subgraphs["Top100"],
                                       feed_dict={self.score_ph: scores})[1]
            top_matches.append(top_matches_val)
        top_matches = np.concatenate(top_matches)
        print("Accuracy using CSLS is")
        self.calc_accuracy(top_matches)
        self.calc_csls(sess)

    def calc_csls(self, sess):
        """
        Calculates the value of CSLS criterion
        Arguments:
            sess(tf.session): Tensorflow session
        """
        good_pairs = self.generate_dictionary(sess)
        eval_dict = {self.src_ph: good_pairs[0],
                     self.tgt_ph: good_pairs[1]}
        cos_mean = sess.run(self.csls_subgraphs["CSLS_Criteria"], feed_dict=eval_dict)
        print("CSLS Score is " + str(cos_mean))

        # Drop LR only after the second drop in CSLS
        if cos_mean < self.best_cos_score:
            self.drop_lr = True & self.second_drop
            self.second_drop = True

        # Save model whenever cos score is better than saved score
        if cos_mean > self.best_cos_score:
            self.save_model = True
        else:
            self.save_model = False

        # Update best cos score
        if cos_mean > self.best_cos_score:
            self.best_cos_score = cos_mean
            self.drop_lr = False

    def get_candidates(self, sess, avg1, avg2, batch_size=512, swap_score=False):
        """
        Get the candidates based on type of dictionary
        Arguments:
             sess(tf.session): Tensorflow session
             dict_type(str): S2T-Source2Target, S2T&T2S (Both)
        Returns:
            Numpy array of max_dict x 2 or smaller
        """
        all_scores = []
        all_targets = []
        for i in range(0, self.max_dict_size, batch_size):
            src_ids = [x for x in range(i, min(i + batch_size, self.max_dict_size))]
            dict_dict = {self.src_ph: src_ids,
                         self.tgt_ph: self.tgt_ids}
            if swap_score:
                temp_score = sess.run(self.csls_subgraphs["ScoreG_T2S"], feed_dict=dict_dict)
            else:
                temp_score = sess.run(self.csls_subgraphs["ScoreGraph"], feed_dict=dict_dict)
            batch_score = 2 * temp_score - (avg1[src_ids][:, None] + avg2[None, :])
            top_matches = sess.run(
                self.csls_subgraphs["Top2"], feed_dict={
                    self.score_ph: batch_score})
            all_scores.append(top_matches[0])
            all_targets.append(top_matches[1])
        all_scores = np.concatenate(all_scores)
        all_targets = np.concatenate(all_targets)
        all_pairs = np.concatenate([
            np.arange(0, self.max_dict_size, dtype=np.int64)[:, None],
            all_targets[:, 0][:, None]
        ], 1)

        # Scores with high confidence will have large difference between first two guesses
        diff = all_scores[:, 0] - all_scores[:, 1]
        reordered = np.argsort(diff, axis=0)
        reordered = reordered[::-1]
        all_pairs = all_pairs[reordered]

        # Select words which are in top max_dict
        selected = np.max(all_pairs, axis=1) <= self.max_dict_size
        all_pairs = all_pairs[selected]

        # Make sure size is less than max_dict
        all_pairs = all_pairs[:self.max_dict_size]
        return all_pairs

    def generate_dictionary(self, sess, dict_type="S2T"):
        """
        Generates best translation pairs
        Arguments:
             sess(tf.session): Tensorflow session
             dict_type(str): S2T-Source2Target, S2T&T2S (Both)
        Returns:
            Numpy array of max_dict x 2 or smaller
        """
        avg1, avg2 = self.calc_avg_dist(sess)
        s2t_dico = self.get_candidates(sess, avg1, avg2)
        print("Completed generating S2T dictionary of size " + str(len(s2t_dico)))
        if dict_type == "S2T":
            map_src_ind = np.asarray([s2t_dico[x][0] for x in range(len(s2t_dico))])
            tra_tgt_ind = np.asarray([s2t_dico[x][1] for x in range(len(s2t_dico))])
            return [map_src_ind, tra_tgt_ind]
        if dict_type == "S2T&T2S":
            # This case we are running Target 2 Source mappings
            t2s_dico = self.get_candidates(sess, avg2, avg1, swap_score=True)
            print("Completed generating T2S dictionary of size " + str(len(t2s_dico)))
            t2s_dico = np.concatenate([t2s_dico[:, 1:], t2s_dico[:, :1]], 1)
            # Find the common pairs between S2T and T2S
            s2t_candi = set([(a, b) for a, b in s2t_dico])
            t2s_candi = set([(a, b) for a, b in t2s_dico])
            final_pairs = s2t_candi & t2s_candi
            dico = np.asarray(list([[a, b] for (a, b) in final_pairs]))
            print("Completed generating final dictionary of size " + str(len(final_pairs)))
            return dico
