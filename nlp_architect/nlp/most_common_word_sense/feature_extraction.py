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
# ****************************************************************************
import nltk
import re
import numpy
from numpy import dot
from numpy.linalg import norm

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

# -------------------------------------------------------------------------------------#


def extract_features_envelope(target_word, definition, hyps_vec, model_w2v):
    valid_w2v_flag, target_word_emb = return_w2v(target_word, model_w2v)

    # calculate target word to definition similarity and definition similarity CBOW
    definition_words = extract_meaningful_words_from_sentence(definition)
    definition_sim = calc_word_to_sentence_sim_w2v(target_word, definition_words, model_w2v, 0)
    definition_sentence_emb_cbow, definition_sim_cbow = \
        calc_word_to_sentence_dist_cbow(target_word, definition_words, model_w2v)

    # calculate hypernyms similarity
    hyps_vec = convert_string_to_list_of_words(hyps_vec)
    hyps_sim = calc_word_to_sentence_sim_w2v(target_word, hyps_vec, model_w2v, 2)

    return [valid_w2v_flag, definition_sim_cbow, definition_sim, hyps_sim, target_word_emb,
            definition_sentence_emb_cbow]


# -------------------------------------------------------------------------------------#
def extract_meaningful_words_from_sentence(sentence):
    sentence = re.sub('[-+.^:,\\[|\\]|\\(|\\)]', '', str(sentence))

    tokens = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)
    words_vec = []
    cntr = 0
    for word, tag in pos_tags:
        if tag.startswith('NN') | tag.startswith('VB'):
            if not bool(re.search(r'\d', word)):  # disqualify words does not contain digits
                words_vec.insert(cntr, word)
                cntr += 1

    return words_vec


# -------------------------------------------------------------------------------------#
def convert_string_to_list_of_words(string_list_of_words):
    string_list_of_words = re.sub('[-+.^:,\\[|\\]|\\(|\\)]', '', str(string_list_of_words))
    tokens = nltk.word_tokenize(string_list_of_words)

    words_vec = []
    cntr = 0
    for word in tokens:
        words_vec.insert(cntr, word)
        cntr += 1

    return words_vec


# -------------------------------------------------------------------------------------#
def calc_word_to_sentence_dist_cbow(target_word, sentence, model):
    cbow_sentence_emb = calc_cbow_sentence(sentence, model)

    try:
        wv_target_word = numpy.array
        wv_target_word = model[target_word]
        cosine_sim = cosine_similarity(wv_target_word, cbow_sentence_emb)
    # if target word is not in embedding dictionary
    except Exception:
        cosine_sim = 0

    return cbow_sentence_emb, cosine_sim


# -------------------------------------------------------------------------------------#

def cosine_similarity(vec1, vec2):

    cosine_sim = 0
    try:
        norm_vec1 = norm(vec1)
        norm_vec2 = norm(vec2)
        den = norm_vec1*norm_vec2
        if den != 0:
            cosine_sim = dot(vec1, vec2) / den
    except Exception:  # if target word is not in embedding dictionary
        cosine_sim = 0

    return cosine_sim
# -------------------------------------------------------------------------------------#


# -------------------------------------------------------------------------------------#
def calc_cbow_sentence(sentence, model):
    cbow_sentence = numpy.zeros(300)
    i = 0
    for word in sentence:
        try:
            wv = numpy.array
            wv = model[word]
            cbow_sentence = cbow_sentence + wv
            i += 1
        except Exception:   # if word is not in embedding dictionary
            pass  # no-operation

    if i > 0:
        cbow_sentence = cbow_sentence / i

    return cbow_sentence


# -------------------------------------------------------------------------------------#
def calc_word_to_sentence_sim_w2v(target_word, string_vec, model, max_items_to_test):
    sim_score_vec = []
    i = 0
    for word in string_vec:
        # remove leading white spaces
        word = word.strip()
        if target_word != word:
            sim_score = w2v_similarity_envelope(target_word, word, model)
            sim_score_vec.insert(i, sim_score)
            i += 1
        if max_items_to_test > 0:
            if max_items_to_test == i:
                break

    top_av = calc_top_av(sim_score_vec)
    return top_av


# -------------------------------------------------------------------------------------#
def w2v_similarity_envelope(word_a, phrase_b, model):
    try:
        similarity = -1
        sim_scores_vec = []
        word_vec = phrase_b.split("_")  # in case wordB is a phrase
        i = 0
        for wordB in word_vec:
            # if (wordA != wordB) & ( wordA not in stopwords.words('english')):
            if word_a != wordB:
                sim = w2v_similarity(word_a, wordB, model)
                if sim > -1:
                    sim_scores_vec.insert(i, sim)
                    i = i + 1
        if len(sim_scores_vec) > 0:
            similarity = numpy.mean(sim_scores_vec)

        return similarity
    except Exception:
        return -1


# -------------------------------------------------------------------------------------#

def w2v_similarity(word_a, word_b, model):
    try:
        similarity = model.similarity(word_a, word_b)
        return similarity
    except Exception:
        return -1


# -------------------------------------------------------------------------------------#

def return_w2v(word_a, model):
    try:
        w2v = numpy.array
        w2v = model[word_a]
        return True, w2v
    except Exception:
        return False, w2v


# -------------------------------------------------------------------------------------#
def calc_top_av(sim_score_vec):
    av_number_th = 3
    sim_score_vec_sorted = sorted(sim_score_vec, reverse=True)
    cntr = 0
    score_acc = 0
    for score in sim_score_vec_sorted:
        if score > -1:
            score_acc = score_acc + score
            cntr = cntr + 1
            if cntr >= av_number_th:
                break

    if cntr > 0:
        av_score = score_acc / cntr
    else:
        av_score = 0

    return av_score


# -------------------------------------------------------------------------------------#

def get_inherited_hypernyms_list(synset, hyps_list):

    for hypernym in synset.hypernyms():
        hyp_string = hypernym.name()
        hyp_string = hyp_string.split(".")[0]
        hyps_list.append(hyp_string)

        hypernym = set(get_inherited_hypernyms_list(hypernym, hyps_list))
    return hyps_list


# -------------------------------------------------------------------------------------#
def get_synonyms(synset):
    synonym_list = []
    i = 0
    for synonym in synset.lemma_names():
        synonym_list.insert(i, synonym.replace('_', ' '))
        i = i + 1

    return synonym_list


# -------------------------------------------------------------------------------------#
def extract_synset_data(synset):
    # a. get definition
    definition = synset.definition()
    # b. get inherited hypernyms
    hyper_list = []
    hyps_list = get_inherited_hypernyms_list(synset, hyper_list)

    # c. get synonyms
    synonym_list = get_synonyms(synset)

    return definition, hyps_list, synonym_list
