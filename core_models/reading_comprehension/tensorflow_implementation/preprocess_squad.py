from __future__ import print_function
from tensorflow.python.platform import gfile
import numpy as np
import nltk
import argparse
import json
import os
from tqdm import *
from utils import sanitize_path

_PAD = b"<pad>"
_SOS = b"<sos>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _SOS, _UNK]
# nltk.download('punkt')


def create_token_map(para, para_tokens):
    token_map = {}
    char_append = ''
    para_token_idx = 0
    for char_idx, char in enumerate(para):
        if char != u' ':
            current_para_token = (para_tokens[para_token_idx])
            char_append += char
            if char_append == current_para_token:
                token_map[
                    char_idx -
                    len(char_append) +
                    1] = [
                    char_append,
                    para_token_idx]
                para_token_idx += 1
                char_append = ''

    return token_map


def create_vocabulary(data_lists, vocabulary_path=None):
    vocab = {}
    for list_element in data_lists:
        for sentence in list_element:
            for word in sentence:
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1

    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    vocab_dict = dict((vocab_ele, i) for i, vocab_ele in enumerate(vocab_list))

    return vocab_list, vocab_dict


def get_glove_matrix(vocab_list, download_path):
    if not gfile.Exists(download_path + ".npz"):
        vocab_len = len(vocab_list)
        glove_path = os.path.join(download_path + "glove.6B.300d.txt")
        glove_matrix = np.zeros((vocab_len, 300))
        count = 0
        with open(glove_path) as f:
            for line in tqdm(f):
                split_line = line.lstrip().rstrip().split(" ")
                word = split_line[0]
                word_vec = list(map(float, split_line[1:]))
                if word in vocab_list:
                    word_index = vocab_list.index(word)
                    glove_matrix[word_index, :] = word_vec
                    count += 1
                if word.capitalize() in vocab_list:
                    word_index = vocab_list.index(word.capitalize())
                    glove_matrix[word_index, :] = word_vec
                    count += 1
                if word.upper() in vocab_list:
                    word_index = vocab_list.index(word.upper())
                    glove_matrix[word_index, :] = word_vec
                    count += 1

        print("Saving the embeddings .npz file")
        save_file_name = download_path + "/glove.trimmed.300d"
        np.savez_compressed(save_file_name, glove=glove_matrix)
        print(
            "Out of %d total words in the vocab, %d words have valid glove vectors",
            vocab_len,
            count)


def tokenize_sentence(line):
    tokenized_words = [
        word.replace(
            "``",
            '"').replace(
            "''",
            '"') for word in nltk.word_tokenize(line)]
    return list(map(lambda x: str(x), tokenized_words))


def extract_data_from_files(json_data):
    data_para = []
    data_ques = []
    data_answer = []
    skipped = 0
    for article_id in range(len(json_data['data'])):
        sub_paragraphs = json_data['data'][article_id]['paragraphs']
        for para_id in range(len(sub_paragraphs)):
            req_para = sub_paragraphs[para_id]['context']
            req_para = req_para.replace("''", '" ')
            req_para = req_para.replace("``", '" ')
            para_tokens = tokenize_sentence(req_para)
            answer_map = create_token_map(req_para, para_tokens)

            questions = sub_paragraphs[para_id]['qas']
            for ques_id in range(len(questions)):

                req_question = questions[ques_id]['question']
                question_tokens = tokenize_sentence(req_question)

                for ans_id in range(1):
                    answer_text = questions[ques_id]['answers'][ans_id]['text']
                    answer_start = questions[ques_id][
                        'answers'][ans_id]['answer_start']
                    answer_end = answer_start + len(answer_text)
                    text_tokens = tokenize_sentence(answer_text)
                    last_word_answer = len(text_tokens[-1])
                    try:
                        a_start_idx = answer_map[answer_start][1]

                        a_end_idx = answer_map[
                            answer_end - last_word_answer][1]

                        data_para.append(para_tokens)
                        data_ques.append(question_tokens)
                        data_answer.append((a_start_idx, a_end_idx))

                    except Exception as e:
                        skipped += 1

    return data_para, data_ques, data_answer


def write_to_file(file_dict, path_to_save):

    for file_name in file_dict:
        with gfile.GFile(os.path.join(path_to_save + file_name), mode='w') as target_file:
            for line in file_dict[file_name]:
                target_file.write(" ".join([str(tok) for tok in line]) + "\n")


def get_ids_list(data_lists, vocab):
    ids_list = []
    for line in data_lists:
        curr_line_idx = []
        for word in line:
            try:
                curr_line_idx.append(vocab[word])
            except:
                curr_line_idx.append(vocab[_UNK])

        ids_list.append(curr_line_idx)
    return ids_list

if __name__ == '__main__':

    # parse the command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--download_path',
        default='',
        help='enter path where training data and the glove embeddings were downloaded')

    parser.add_argument(
        '--preprocess_glove',
        type=int,
        default=1,
        help='Chose whether or not to preprocess glove embeddings')

    parser.set_defaults()

    args = parser.parse_args()
    glove_flag = args.preprocess_glove

    try:
        assert os.path.exists(args.download_path)
    except:
        print("Please enter a valid download path")
        exit()

    data_path = sanitize_path(args.download_path)

    # Load Train and Dev Data
    train_filename = os.path.join(data_path + "/train-v1.1.json")
    dev_filename = os.path.join(data_path + "/dev-v1.1.json")
    with open(train_filename) as train_file:
        train_data = json.load(train_file)

    with open(dev_filename) as dev_file:
        dev_data = json.load(dev_file)

    # Extract training data from raw files
    train_para, train_question, train_ans = extract_data_from_files(train_data)
    # Extract dev data from raw dataset
    dev_para, dev_question, dev_ans = extract_data_from_files(dev_data)

    data_lists = [train_para, train_question, dev_para, dev_question]
    # Obtain vocab list
    print("Create Vocabulary")
    vocab_list, vocab_dict = create_vocabulary(data_lists)

    # Obtain embedding matrix from pre-trained glove vectors:
    if glove_flag:
        print("Preprocessing glove")
        get_glove_matrix(vocab_list, data_path)

    # Get vocab ids file
    train_para_ids = get_ids_list(train_para, vocab_dict)
    train_question_ids = get_ids_list(train_question, vocab_dict)
    dev_para_ids = get_ids_list(dev_para, vocab_dict)
    dev_question_ids = get_ids_list(dev_question, vocab_dict)

    final_data_dict = {"train.ids.context": train_para_ids,
                       "train.ids.question": train_question_ids,
                       "val.ids.context": dev_para_ids,
                       "val.ids.question": dev_question_ids}

    print("writing to file")
    write_to_file(final_data_dict, data_path)
