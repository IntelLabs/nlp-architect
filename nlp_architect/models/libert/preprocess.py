import csv
import os
from pathlib import Path
from itertools import permutations
from transformers import BertTokenizer
import numpy as np
from tqdm import tqdm
import spacy
from spacy.tokens import Doc

DATA_DIR = Path(os.path.realpath(__file__)).parent / 'data' / 'conll'

class SpacyWithBertTokenizer:
    def __init__(self):
        self.parse_cache = {}
        self.nlp = self.get_spacy_with_bert_tokenizer()

    @staticmethod
    def get_spacy_with_bert_tokenizer(spacy_model="en_core_web_lg"):
        nlp = spacy.load(spacy_model, disable=["ner", "vectors", "textcat"])
        bert_tokenizer_m = ModifiedBertTokenizer.from_pretrained('bert-base-uncased')
        nlp.tokenizer = Tokenizer(nlp.vocab, bert_tokenizer_m)
        return nlp

    def pipe_batch(self, batch):
        cached_idx = {}
        not_in_cache_idx = {}
        not_in_cache = []
        for i, text in enumerate(batch):
            if text in self.parse_cache:
                cached_idx[i] = self.parse_cache[text]
            else:
                not_in_cache_idx[i] = len(not_in_cache)
                not_in_cache.append(text)

        parsed_docs = list(self.nlp.pipe(not_in_cache))
        res = []
        for i in range(len(batch)):
            if i in cached_idx:
                res.append(cached_idx[i])
            else:
                parsed_doc = parsed_docs[not_in_cache_idx[i]]
                self.parse_cache[batch[i]] = parsed_doc
                res.append(parsed_doc)
        return res

    def pipe(self, word_lists):
        docs = []
        for s_batch in tqdm(batcher([' '.join(words) for words in word_lists], n=20)):
            docs.extend(self.pipe_batch(s_batch))

        res_list = []
        for doc in docs:
            sub_toks = [''.join(sub_tok) if len(sub_tok) > 1 else '' for sub_tok in doc.user_data]
            toks = [t.text for t in doc]
            pos = [t.tag_ for t in doc]
            heads = [t.head.i for t in doc]
            rels = [t.dep_ for t in doc]
            assert len(toks) == len(sub_toks) == len(pos) == len(heads) == len(rels)
            res_list.append((toks, pos, sub_toks, heads, rels))
        return res_list

def batcher(iterable, n=1):
    iter_len = len(iterable)
    for ndx in range(0, iter_len, n):
        yield iterable[ndx: min(ndx + n, iter_len)]

def conll_iter(f):
    toks, labels = [], []
    for line in f:
        line = line.strip()
        if not line:
            yield tuple(toks), tuple(labels)
            toks, labels = [], []
        else:
            split = line.split('\t')
            labels.append(split[1])
            toks.append(split[0])

def parse_file(txt_path, spacy_bert_tok, overwrite=True):
    out_path = txt_path[:-4].replace('conll', 'csv') + '.csv'
    os.makedirs(Path(out_path).parent, exist_ok=True)

    if overwrite or not os.path.exists(out_path):
        with open(txt_path) as input_f:
            space_tok_reviews = []
            reviews_tok_labels = []
            for toks, labels in conll_iter(input_f):
                space_tok_reviews.append(toks)
                reviews_tok_labels.append(labels)

            with open(out_path, 'w') as csv_file:
                writer = csv.writer(csv_file)
                parsed_reviews = spacy_bert_tok.pipe(space_tok_reviews)
                for word_list, parsed_review, labels in \
                    zip(space_tok_reviews, parsed_reviews, reviews_tok_labels):
                    toks = parsed_review[0]
                    assert len(toks) == len(labels)
                    assert tuple(toks) == word_list
                    for tok, pos_tag, sub_tok, head, rel, label in zip(*parsed_review, labels):
                        writer.writerow([tok, label, head, rel, pos_tag, sub_tok])
                    writer.writerow(['_'] * 6)
                    writer.writerow(['_'] * 6)
                    # binary_heads = add_sub_tokens(binarize(heads), bert_sub_toks)
                    # assert binary_heads.shape[0] == binary_heads.shape[1]
                    # heads_list.append(binary_heads)
                # np.savez(txt_path[:-4] + '_spacy_heads.npz', *heads_list)

def add_sub_tokens(preds, sub_tokens, zero_sub_tokens=False):
    for i in range(len(sub_tokens) - 1, -1, -1):
        for _ in range(1, len(sub_tokens[i])):
            if zero_sub_tokens:
                sub_token_row = [0 for _ in preds[i]]
            else:
                sub_token_row = preds[i].copy()
            preds.insert(i + 1, sub_token_row)

    for row in preds:
        # zero [CLS] (ROOT) column?
        for i in range(len(sub_tokens) - 1, -1, -1):
            for _ in range(1, len(sub_tokens[i])):
                row.insert(i + 2, 0)

    # insert zeros row for [CLS] token
    preds.insert(0, [0] * (len(preds) + 1))
    return np.array(preds)

def binarize(preds):
    res = []
    for pred in preds:
        res.append([1 if i == pred else 0 for i in range(len(preds) + 1)])
    return res

class ModifiedBertTokenizer(BertTokenizer):
    def __init__(self, *args, **kwargs):
        # kwargs['do_lower_case'] = False
        super(ModifiedBertTokenizer, self).__init__(*args, **kwargs)

    def tokenize(self, text, **kwargs):
        """ Converts a string in a sequence of tokens (string), using the tokenizer.
            Split in words for word-based vocabulary or sub-words for sub-word-based
            vocabularies (BPE/SentencePieces/WordPieces).

            Take care of added tokens.
        """
        def split_on_tokens(tok_list, text, res_toks=None):
            if not text:
                return []
            if not tok_list:
                return self._tokenize(text, **kwargs)
            tok = tok_list[0]
            split_text = text.split(tok)
            if res_toks is not None:
                res_toks.extend(split_text)
            sub_tokens = sum((split_on_tokens(tok_list[1:], sub_text.strip()) + [tok] \
                        for sub_text in split_text), [])[:-1]
            return sub_tokens

        added_tokens = list(self.added_tokens_encoder.keys()) + self.all_special_tokens
        tokens = []
        sub_tokens = split_on_tokens(added_tokens, text, tokens)
        return tokens, sub_tokens

class Tokenizer:
    def __init__(self, vocab, bert_tokenizer):
        self.vocab = vocab
        self.bert_tokenizer = bert_tokenizer

    def __call__(self, text):
        res = []
        words = text.split()
        sub_tokens = []
        for word in words:
            bert_token, sub_tok = self.bert_tokenizer.tokenize(word)
            sub_tokens.append(sub_tok)
            res.extend(bert_token)
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        doc = Doc(vocab=self.vocab, words=res, spaces=spaces)
        doc.user_data = sub_tokens
        return doc

def parse_cross_domain_settings(domains: list, overwrite=True):
    spacy_bert_tok = SpacyWithBertTokenizer()

    for domain_a, domain_b in permutations(domains, r=2):
        print('Setting: ' + domain_a + ' to ' + domain_b)
        for split in ['1', '2', '3']:
            for ds in tqdm(['train', 'test', 'dev']):
                ds_file = str(DATA_DIR / f'{domain_a}_to_{domain_b}_{split}' / f'{ds}.txt')
                print(ds_file)
                parse_file(ds_file, spacy_bert_tok, overwrite=overwrite)

if __name__ == "__main__":
    parse_cross_domain_settings(domains=['restaurants', 'laptops', 'device'], overwrite=False)
