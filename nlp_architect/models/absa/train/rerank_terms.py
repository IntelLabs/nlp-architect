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
import csv
import pickle
import numpy as np
import tensorflow
from os import PathLike


from nlp_architect.models.absa.utils import _read_generic_lex_for_similarity
from nlp_architect.models.absa import TRAIN_OUT, TRAIN_LEXICONS

from scipy.spatial.distance import cosine
from sklearn.model_selection import StratifiedKFold
# pylint: disable=import-error
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, load_model

from nlp_architect.utils.embedding import load_word_embeddings


class RerankTerms(object):
    out_dir = TRAIN_OUT / 'output'
    model_dir = TRAIN_OUT / 'reranking_model'
    train_rerank_data_path = TRAIN_LEXICONS / 'RerankTrainingData.csv'
    PREDICTION_THRESHOLD = 0.7

    def __init__(self, vector_cache=True, rerank_model: PathLike = None,
                 emb_model_path: PathLike = None):
        # model and training params
        self.embeddings_len = 300
        self.activation_1 = 'relu'
        self.activation_2 = 'relu'
        self.activation_3 = 'sigmoid'
        self.loss = 'binary_crossentropy'
        self.optimizer = 'rmsprop'

        self.epochs_and_batch_size = [(10, 2)]
        self.seeds = [3]
        self.threshold = 0.5

        self.sim_lexicon = TRAIN_LEXICONS / 'RerankSentSimLex.csv'
        self.generic_lexicon = TRAIN_LEXICONS / 'GenericOpinionLex.csv'

        self.vector_cache = vector_cache
        self.word_vectors_dict = {}
        self.vectors_sim_dict = {}

        self.rerank_model_path = rerank_model
        self.emb_model_path = emb_model_path

        RerankTerms.out_dir.mkdir(parents=True, exist_ok=True)

        tensorflow.logging.set_verbosity(tensorflow.logging.ERROR)

    def calc_cosine_similarity(self, word_1, word_2, embedding_dict):
        """
        calculate cosine similarity scores between 2 terms

        Args:
            word_1 (str): 1st input word
            word_2 (str): 2nd input word
            embedding_dict (dict): embedding dictionary

        Returns:
            vectors_sim_dict[key] (float): similarity scores between the 2 input words

        """
        key = str(sorted([word_1, word_2]))

        if not self.vector_cache or key not in self.vectors_sim_dict:
            vector_1 = embedding_dict.get(word_1)
            vector_2 = embedding_dict.get(word_2)

            # check if both words have vectors
            if np.count_nonzero(vector_1) > 0 and np.count_nonzero(vector_2) > 0:
                sim_score = cosine(vector_1, vector_2)
            else:
                sim_score = None
            self.vectors_sim_dict[key] = sim_score

        return self.vectors_sim_dict[key]

    def calc_similarity_scores_for_all_terms(self, terms, generic_terms, embedding_dict):
        """
        calculate similarity scores between each term and each off the generic terms

        Args:
            terms: candidate terms
            generic_terms: generic opinion terms
            embedding_dict: embedding dictionary

        Returns:
            neg_all: similarity scores between each cand term and neg generic term
            pos_all: similarity scores between each cand term and pos generic term

        """
        print("\nComputing similarity scores...\n")

        neg_all = []
        pos_all = []

        for term in terms:
            polarity_sim_dic = {'NEG': [], 'POS': []}
            for generic_term, polarity in generic_terms.items():

                sim_score = self.calc_cosine_similarity(term, generic_term, embedding_dict)

                if sim_score is not None:
                    polarity_sim_dic[polarity].append(sim_score)
                else:
                    polarity_sim_dic[polarity].append(float(0))

            neg_all.append(polarity_sim_dic['NEG'])
            pos_all.append(polarity_sim_dic['POS'])

        return neg_all, pos_all

    @staticmethod
    def load_terms_and_polarities(filename):
        """
        load terms and polarities from file

        Args:
            filename: feature table file full path

        Returns:
            terms: candidate terms
            polarities: opinion polarity per term

        """
        print('Loading training data from {} ...'.format(filename))

        table = np.genfromtxt(filename, delimiter=',', skip_header=1, dtype=str)
        if table.size == 0:
            raise ValueError('Error: Term file is empty, no terms to re-rank.')

        try:
            terms = table[:, 1]
        except Exception as e:
            print("\n\nError converting str to float in training table: {}".format(e))

        polarities = table[:, 3].astype(str)

        if len(terms) != len(polarities):
            raise ValueError(
                'Count of opinion terms is different than the count of loaded polarities.')
        polarities = {terms[i]: polarities[i] for i in range(len(terms))}

        print(str(terms.shape[0]) + ' features loaded from CSV file')
        return terms, polarities

    @staticmethod
    def load_terms_and_y_labels(filename):
        """Load terms and Y labels from feature file.

        Args:
            filename: feature table file full path

        Returns:
            x: feature vector
            y: labels vector
            terms: candidate terms
            polarities: opinion polarity per term
        """
        print('Loading basic features from {} ...'.format(filename))

        table = np.genfromtxt(filename, delimiter=',', skip_header=1, dtype=str)
        if table.size == 0:
            raise ValueError('Error: Terms file is empty, no terms to re-rank.')

        try:
            terms = table[:, 1]
        except Exception as e:
            print("\n\nError converting str to float in training table: {}".format(e))

        y = table[:, 0].astype(int)
        polarities = None

        print(str(terms.shape[0]) + ' features loaded from CSV file')
        return y, terms, polarities

    @staticmethod
    def concat_sim_scores_and_features(x, neg_sim, pos_sim):
        """
        concatenate similarity scores to features

        Args:
            x: feature vector
            neg_sim: similarity scores between cand terms and neg opinion terms
            pos_sim: similarity scores between cand terms and pos opinion terms

        Returns:
            x: concatenated features and similarity scores
        """
        neg = np.array(neg_sim)
        pos = np.array(pos_sim)

        neg_avg = np.mean(neg, axis=1, keepdims=True)
        neg_std = np.std(neg, axis=1, keepdims=True)
        neg_min = np.min(neg, axis=1, keepdims=True)
        neg_max = np.max(neg, axis=1, keepdims=True)

        pos_avg = np.mean(pos, axis=1, keepdims=True)
        pos_std = np.std(pos, axis=1, keepdims=True)
        pos_min = np.min(pos, axis=1, keepdims=True)
        pos_max = np.max(pos, axis=1, keepdims=True)

        print('\nAdding polarity similarity features...')

        res_x = np.concatenate(
            (neg_avg, neg_std, neg_min, neg_max, pos_avg, pos_std, pos_min, pos_max, x), 1)

        return res_x

    def generate_embbeding_features(self, terms, embedding_dict):
        """
        concatenate word embedding to features

        Args:
            terms: candidate terms
            embedding_dict: embedding dictionary
            word_to_emb_idx: index to embedding dictionary
        Returns:
            x: concatenated features and word embs
        """
        print("\nAdding word vector features...\n")
        vec_matrix = np.zeros((len(terms), self.embeddings_len))

        j = 0
        for term in terms:
            word_vector = embedding_dict.get(term)
            vec_matrix[j, :] = word_vector
            j += 1

        x = vec_matrix[:j]

        return x

    def load_terms_and_y_labels_and_generate_features(self, filename):
        """
       load candidate terms with their basic features, Y labels and polarities from feature file

       Args:
           filename: feature table file path
       Returns:
           x: feature vector
           y: labels vector
           terms: candidate terms
           polarities: opinion polarity per term
       """
        print("\nLoading feature table...\n")

        y, terms, polarities = self.load_terms_and_y_labels(filename)

        x, terms, polarities = self.generate_features(terms, polarities)

        y_vector = None
        if y is not None:
            y_vector = np.reshape(y, (y.shape[0], 1))

        return x, y, y_vector, terms, polarities

    def load_terms_and_generate_features(self, filename):
        """
       load candidate terms with their basic features, Y labels and polarities from feature file

       Args:
           filename: feature table file path
       Returns:
           x: feature vector
           terms: candidate terms
           polarities: opinion polarity per term
       """
        print("\nLoading feature table...\n")

        terms, polarities = self.load_terms_and_polarities(filename)

        x, terms, polarities = self.generate_features(terms, polarities)

        return x, terms, polarities

    @staticmethod
    def _determine_unk_polarities(terms, polarities, neg, pos):

        for i, term in enumerate(terms):
            if np.average(pos[i]) <= np.average(neg[i]):
                polarities[term] = 'POS'
            else:
                polarities[term] = 'NEG'

        return polarities

    def generate_features(self, terms, polarities):

        generic_terms = _read_generic_lex_for_similarity(self.generic_lexicon)
        # generate unified list of candidate terms and generic terms
        terms_list = [term for term in terms]
        for term in generic_terms.keys():
            terms_list.append(term.strip('\'"'))

        print("\nLoading embedding model...\n")
        embedding_dict, _ = load_word_embeddings(self.emb_model_path, terms_list)
        x = self.generate_embbeding_features(terms, embedding_dict)
        neg, pos = self.calc_similarity_scores_for_all_terms(terms, generic_terms, embedding_dict)
        x = self.concat_sim_scores_and_features(x, neg, pos)
        polarities = self._determine_unk_polarities(terms, polarities, neg, pos)
        print("\nDimensions of X: " + str(x.shape))
        return x, terms, polarities

    def evaluate(self, model, x_test, y_test, terms):
        report = {}
        predictions = model.predict(x_test, verbose=0)
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for i, prediction in enumerate(predictions):
            y_true = y_test[i][0]

            if prediction[0] > self.threshold:
                y_pred = 1
            else:
                y_pred = 0

            report[terms[i]] = (prediction[0], y_pred, y_true)

            if y_pred == 1:
                if y_true == 1:
                    tp = tp + 1
                else:
                    fp = fp + 1
            elif y_true == 0:
                tn = tn + 1
            else:
                fn = fn + 1

        prec = 100 * tp / (tp + fp)
        rec = 100 * tp / (tp + fn)
        f1 = 2 * (prec * rec) / (prec + rec)

        return (prec, rec, f1), report

    def generate_model(self, input_vector_dimension):
        """Generate MLP model.

        Args:
           input_vector_dimension (int): word emb vec length

        Returns:
        """
        mlp_model = Sequential()

        mlp_model.add(Dense(128, activation=self.activation_1, input_dim=input_vector_dimension))
        mlp_model.add(Dropout(0.5))
        mlp_model.add(Dense(64, activation=self.activation_2))
        mlp_model.add(Dropout(0.5))
        mlp_model.add(Dense(1, activation=self.activation_3))
        mlp_model.compile(metrics=['accuracy'], loss=self.loss, optimizer=self.optimizer)

        return mlp_model

    def predict(self, input_table_file, generic_opinion_terms):
        """Predict classification class according to model.

        Args:
           input_table_file: feature(X) and labels(Y) table file
           generic_opinion_terms: generic opinion terms file name

        Returns:
            final_concat_opinion_lex: reranked_lex conctenated with generic lex
        """
        x, terms, polarities = self.load_terms_and_generate_features(input_table_file)

        model = load_model(self.rerank_model_path)
        reranked_lexicon = model.predict(x, verbose=0)

        reranked_lex = {}
        for i, prediction in enumerate(reranked_lexicon):
            if not np.isnan(prediction[0]) and prediction[0] > self.PREDICTION_THRESHOLD:
                reranked_lex[terms[i]] = (prediction[0], polarities[terms[i]])

        final_concat_opinion_lex = \
            self._generate_concat_reranked_lex(reranked_lex, generic_opinion_terms)
        self._write_prediction_results(final_concat_opinion_lex)
        return final_concat_opinion_lex

    def rerank_train(self):
        """Class for training a reranking model."""
        x, y, _, _, _ = \
            self.load_terms_and_y_labels_and_generate_features(self.train_rerank_data_path)

        try:
            print('\nModel training...')
            model = self.generate_model(x.shape[1])
            e = self.epochs_and_batch_size[0][0]
            b = self.epochs_and_batch_size[0][1]

            model.fit(x, y, epochs=e, batch_size=b, verbose=0)
            RerankTerms.model_dir.mkdir(parents=True, exist_ok=True)

            model.save(str(RerankTerms.model_dir) + '/rerank_model.h5')
            print('\nSaved model to: ' + str(RerankTerms.model_dir) + '/rerank_model.h5')

        except ZeroDivisionError:
            print("Division by zero, skipping test")

    def cross_validation_training(self, verbose=False):
        """Perform k fold cross validation and evaluate the results."""
        final_report = {}
        x, y, y_vector, terms, _ = \
            self.load_terms_and_y_labels_and_generate_features(self.train_rerank_data_path)

        for seed in self.seeds:
            np.random.seed(seed)

            for epochs, batch_size in self.epochs_and_batch_size:
                self.print_params(batch_size, epochs, seed)
                k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                f1_scores = []
                precision_scores = []
                recall_scores = []

                try:

                    for i, (train, test) in enumerate(k_fold.split(x, y)):
                        model = self.generate_model(x.shape[1])
                        model.fit(x[train], y_vector[train], epochs=epochs, batch_size=batch_size,
                                  verbose=0)

                        measures, report = self.evaluate(model, x[test], y_vector[test],
                                                         terms[test])
                        final_report.update(report)

                        precision, recall, f1 = measures
                        f1_scores.append(f1)
                        precision_scores.append(precision)
                        recall_scores.append(recall)

                        if verbose:
                            print("Fold " + str(i + 1) + ":")
                            self.print_evaluation_results(precision, recall, f1)

                    print('\nSummary:')
                    self.print_evaluation_results(precision_scores, recall_scores, f1_scores)

                except ZeroDivisionError:
                    print("Division by zero, skipping test")

        self.write_evaluation_report(final_report)

    def print_params(self, batch_size, epochs, seed):
        """Print training params.

        Args:
            batch_size(int): batch size
            epochs(int): num of epochs
            seed(int): seed
        """
        print('\nModel Parameters: act_1= ' + self.activation_1 + ', act_2= ' + self.activation_2
              + ', act_3= ' + self.activation_3 + ', loss= ' + self.loss + ', optimizer= '
              + self.optimizer + '\nseed= ' + str(seed) + ', epochs= ' + str(epochs)
              + ', batch_size= ' + str(batch_size) + ', threshold= ' + str(self.threshold)
              + ', use_complete_w2v= ' + ', sim_lexicon= ' + str(self.sim_lexicon) + '\n')

    def print_evaluation_results(self, precision, recall, f1):
        """Print evaluation results.

        Args:
            precision(list of float): precision
            recall(list of float): recall
            f1(list of float): f measure
        """
        print()
        self.print_measure('Precision', precision)
        self.print_measure('Recall', recall)
        self.print_measure('F-measure', f1)
        print('-------------------------------------------------------------------------'
              '------------------------------')

    @staticmethod
    def print_measure(measure, value):
        """Print single measure.

        Args:
            measure(str): measure type
            value(list of float): value
        """
        print(measure + ': {:.2f}%'.format(np.mean(value)), end='')
        if not np.isscalar(value):
            print(" (+/- {:.2f}%)".format(np.std(value)), end='')
        print()

    @staticmethod
    def _generate_concat_reranked_lex(acquired_opinion_lex, generic_opinion_lex_file):
        print('Loading generic sentiment terms from {}...'.format(generic_opinion_lex_file))
        generics_table = np.genfromtxt(generic_opinion_lex_file, delimiter=',', skip_header=1,
                                       dtype=str)
        print(str(generics_table.shape[0]) + ' generic sentiment terms loaded')

        concat_opinion_dict = {}

        for key, value in acquired_opinion_lex.items():
            concat_opinion_dict[key] = (value[0], value[1], 'Y')
        for row in generics_table:
            concat_opinion_dict[row[0]] = (row[2], row[1], 'N')

        return concat_opinion_dict

    @staticmethod
    def _write_prediction_results(concat_opinion_dict):

        out_path = RerankTerms.out_dir / 'generated_opinion_lex_reranked.csv'
        with open(out_path, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Term', 'Score', 'Polarity', 'isAcquired'])
            for key, value in concat_opinion_dict.items():
                writer.writerow([key, value[0], value[1], value[2]])
        print('Reranked opinion lexicon written to {}'.format(out_path))

    @staticmethod
    def write_evaluation_report(report_dic):
        RerankTerms.model_dir.mkdir(parents=True, exist_ok=True)
        out_path = RerankTerms.model_dir / 'rerank_classifier_results.csv'
        with open(out_path, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['term', 'score', 'y_pred', 'y_true'])
            for key, value in report_dic.items():
                writer.writerow([key, value[0], value[1], value[2]])
        print('Report written to {}' + str(out_path))

    @staticmethod
    def load_word_vectors_dict():
        try:
            with open(RerankTerms.model_dir / 'word_vectors_dict.pickle', 'rb') as f:
                ret = pickle.load(f)
        except OSError:
            ret = {}
        return ret
