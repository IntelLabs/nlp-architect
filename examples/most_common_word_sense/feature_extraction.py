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
import re

import nltk
import numpy
from numpy import dot
from numpy.linalg import norm

from nlp_architect.utils.generic import license_prompt

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    if license_prompt('Averaged Perceptron Tagger', 'http://www.nltk.org/nltk_data/') is False:
        raise Exception("can't continue data prepare process "
                        "without downloading averaged_perceptron_tagger")
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    if license_prompt('Punkt model', 'http://www.nltk.org/nltk_data/') is False:
        raise Exception("can't continue data prepare process "
                        "without downloading punkt")
    nltk.download('punkt')


def extract_features_envelope(target_word, definition, hyps_vec, model_w2v):
    """
    extract features

    Args:
        target_word (list(str)): target word for finding senses
        definition (list(str)):  definition of target word
        hyps_vec (list(str)):    hypernym list of of target word
        model_w2v (list(str)):   word embedding's model

    Returns:
        valid_w2v_flag(bool):         true if target word has w2v entry, else false
        definition_sim_cbow(float):   cosine similarity between target word embedding and
        definition cbow sentence embedding
        definition_sim(float):        cosine similarity between target word embedding and
        definition sentence embedding
        hyps_sim(float):              cosine similarity between target word embedding and
        hypernyms embeddings
        target_word_emb(numpy.array): word embedding of target word
        definition_sentence_emb_cbow(numpy.array): definition sentence cbow embedding

    """
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


def extract_meaningful_words_from_sentence(sentence):
    """
    extract meaningful (nouns and verbs) words from sentence

    Args:
        sentence(str): input sentence

    Returns:
        list(str): vector of meaningful words

    """
    sentence = re.sub(r'[-+.^:,\[\]()]', '', str(sentence))

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


def convert_string_to_list_of_words(string_list_of_words):
    """
    convert string to list of words

    Args:
        string_list_of_words(str): input sentence

    Returns:
        list(str): vector of words

    """
    string_list_of_words = re.sub(r'[-+.^:,\[\]()]', '', str(string_list_of_words))
    tokens = nltk.word_tokenize(string_list_of_words)

    words_vec = []
    cntr = 0
    for word in tokens:
        words_vec.insert(cntr, word)
        cntr += 1

    return words_vec


def calc_word_to_sentence_dist_cbow(target_word, sentence, model):
    """
    calculate cosine similaity between word emb. and sentence cbow emb.

    Args:
        target_word(str): input word
        sentence(list(str)): input sentence
        model(gensim.models.Word2Vec): w2v model

    Returns:
        numpy.array: cbow_sentence_emb, sentence cbow embedding
        float: cosine_sim, cosine similarity between target word embedding and cbow sentence
         embedding

    """
    cbow_sentence_emb = calc_cbow_sentence(sentence, model)

    try:
        wv_target_word = model[target_word]
        cosine_sim = cosine_similarity(wv_target_word, cbow_sentence_emb)
    # if target word is not in embedding dictionary
    except KeyError:
        cosine_sim = 0

    return cbow_sentence_emb, cosine_sim


def cosine_similarity(vec1, vec2):
    """
    calculate cosine similarity between 2 vecs

    Args:
        vec1(numpy.array): input vec 1
        vec2(numpy.array): input vec 2

    Returns:
        float: cosine_sim, cosine similarity between 2 vecs


    """

    cosine_sim = 0
    try:
        norm_vec1 = norm(vec1)
        norm_vec2 = norm(vec2)
        den = norm_vec1 * norm_vec2
        if den != 0:
            cosine_sim = dot(vec1, vec2) / den
    except ValueError:  # if target word is not in embedding dictionary
        cosine_sim = 0

    return cosine_sim


def calc_cbow_sentence(sentence, model):
    """
    calc cbow embedding of an input sentence

    Args:
        sentence(bytearray): vector of words
        model(gensim.models.Word2Vec): w2v model

    Returns:
        numpy.array: cbow_sentence, cbow embedding of the input sentence

    """
    cbow_sentence = numpy.zeros(300)
    i = 0
    for word in sentence:
        try:
            wv = model[word]
            cbow_sentence = cbow_sentence + wv
            i += 1
        except KeyError:  # if word is not in embedding dictionary
            pass  # no-operation
        except IndexError:  # if word is not in embedding dictionary
            pass  # no-operation

    if i > 0:
        cbow_sentence = cbow_sentence / i

    return cbow_sentence


def calc_word_to_sentence_sim_w2v(target_word, string_vec, model, max_items_to_test):
    """

    Args:
        target_word(str): input target word
        string_vec(list(str)): sentence
        model(gensim.models.Word2Vec): w2v model
        max_items_to_test(int): max number of words

    Returns:
        float: top_av, average embedding similarity betewwen target word and words in string vec
        (sentence)

    """
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


def w2v_similarity_envelope(word_a, phrase_b, model):
    """
    calculate cosine similarity between 2 words

    Args:
        word_a(str):   input word a
        phrase_b(str): input word b
        model(gensim.models.Word2Vec): w2v model

    Returns:
        float: similarity, mean cosine similarity between 2 words


    """
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
        if sim_scores_vec:
            similarity = numpy.mean(sim_scores_vec)

        return similarity
    except ValueError:
        return -1


def w2v_similarity(word_a, word_b, model):
    """

    Args:
        word_a(str):   input word a
        word_b(str):   input word b (can be phrase)
        model(gensim.models.Word2Vec): w2v model

    Returns:
        float: similarity, cosine similarity between 2 words

    """
    try:
        similarity = model.similarity(word_a, word_b)
        return similarity
    except ValueError:
        return -1
    except IndexError:
        return -1
    except KeyError:
        return -1


def return_w2v(word_a, model):
    """
    extract embedding vector of word_a

    Args:
         word_a(str):   input word
        model(gensim.models.Word2Vec): w2v model

    Returns:
        numpy.array: w2v, embedding vector of word_a

    """
    w2v = None
    try:
        w2v = numpy.array
        w2v = model[word_a]
        return True, w2v
    except ValueError:
        return False, w2v
    except IndexError:
        return False, w2v
    except KeyError:
        return False, w2v


def calc_top_av(sim_score_vec):
    """
    calc top average of scores vector

    Args:
        sim_score_vec(list(float)): vector of similarity scores

    Returns:
        float: av_score, average of top similarity scores in sim_score_vec

    """
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


def get_inherited_hypernyms_list(synset, hyps_list):
    """
    get inherited hypernyms list of synset
    Args:
        synset(synset): synset
        hyps_list: hypernym list

    Returns:
        list(str): hyps_list, hypernym list

    """

    for hypernym in synset.hypernyms():
        hyp_string = hypernym.name()
        hyp_string = hyp_string.split(".")[0]
        hyps_list.append(hyp_string)
        set(get_inherited_hypernyms_list(hypernym, hyps_list))

    return hyps_list


def get_synonyms(synset):
    """
    get synonyms list of synset
    Args:
        synset(synset): synset

    Returns:
        list(str): synonym_list, synonyms list

    """
    synonym_list = []
    i = 0
    for synonym in synset.lemma_names():
        synonym_list.insert(i, synonym.replace('_', ' '))
        i = i + 1

    return synonym_list


def extract_synset_data(synset):
    """

    Args:
        synset(synset): synset

    Returns:
        definition(str): definition of synset
        list(str): synonym_list, synonyms list
        list(str): hyps_list, hypernym list

    """
    # a. get definition
    definition = synset.definition()
    # b. get inherited hypernyms
    hyper_list = []
    hyps_list = get_inherited_hypernyms_list(synset, hyper_list)

    # c. get synonyms
    synonym_list = get_synonyms(synset)

    return definition, hyps_list, synonym_list
