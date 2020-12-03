import csv
import os
from pathlib import Path
from itertools import permutations
from transformers import BertTokenizer
from tqdm import tqdm
import spacy
from spacy.tokens import Doc
import urllib
import pandas as pd
import glob

LIBERT_DIR = Path(os.path.realpath(__file__)).parent
DATA_DIR = LIBERT_DIR / 'data'
CONLL_DIR = DATA_DIR / 'conll'
DOMAINS_DIR = CONLL_DIR / 'domains_all'

class SpacyWithBertTokenizer:
    def __init__(self):
        self.parse_cache = {}
        self.nlp = self.get_spacy_with_bert_tokenizer()
        #  For UD dependency parses, first download a SpaCy UD model- run:
        # > pip install https://storage.googleapis.com/en_ud_model/en_ud_model_lg-1.1.0.tar.gz
        #  Then, replace default `spacy_model` with "en_ud_model_lg" - replace the last line with:
        # self.nlp = self.get_spacy_with_bert_tokenizer(spacy_model="en_ud_model_lg")
        
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
            sub_toks = [' '.join(sub_tok) if len(sub_tok) > 1 else '' for sub_tok in doc.user_data]
            toks = [t.text for t in doc]
            pos = [t.tag_ for t in doc]
            heads = [t.head.i for t in doc]
            head_words = [t.head.text for t in doc]
            rels = [t.dep_ for t in doc]
            assert len(toks) == len(sub_toks) == len(pos) == len(heads) == len(rels)
            res_list.append((toks, pos, sub_toks, heads, head_words, rels))
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
        with open(txt_path, encoding='utf-8') as input_f:
            space_tok_reviews = []
            reviews_tok_labels = []
            for toks, labels in conll_iter(input_f):
                space_tok_reviews.append(toks)
                reviews_tok_labels.append(labels)

            with open(out_path, 'w', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)
                parsed_reviews = spacy_bert_tok.pipe(space_tok_reviews)
                writer.writerow(['TOKEN', 'LABEL', 'HEAD', 'HEAD_WORD', 'DEP_REL', 'POS', 'SUB_TOKENS'])

                for word_list, parsed_review, labels in \
                    zip(space_tok_reviews, parsed_reviews, reviews_tok_labels):
                    toks = parsed_review[0]
                    assert len(toks) == len(labels)
                    assert tuple(toks) == word_list
                    for tok, pos_tag, sub_tok, head, head_word, rel, label in zip(*parsed_review, labels):
                        writer.writerow([tok, label, head, head_word, rel, pos_tag, sub_tok])
                    writer.writerow(['_'] * 7)
                    writer.writerow(['_'] * 7)

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
                tokenized = self._tokenize(text, **kwargs)
                return tokenized
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

def parse_cross_domain(domains: list, splits=3, modes=('train', 'dev', 'test'), overwrite=True):
    spacy_bert_tok = SpacyWithBertTokenizer()

    for domain_a, domain_b in permutations(domains, r=2):
        print('Setting: ' + domain_a + ' to ' + domain_b)
        for split in range(splits):
            for ds in tqdm(modes):
                ds_file = str(CONLL_DIR / f'{domain_a}_to_{domain_b}_{split + 1}' / f'{ds}.txt')
                print(ds_file)
                parse_file(ds_file, spacy_bert_tok, overwrite=overwrite)

def parse_in_domain(domains: list):
    spacy_bert_tok = SpacyWithBertTokenizer()

    for domain in tqdm(domains):
        ds_file = str(DOMAINS_DIR / f'{domain}.txt')
        print('File: ' + ds_file)
        parse_file(ds_file, spacy_bert_tok)

def add_rel_group_column():
    clear_nlp_readme_url = \
    'https://raw.githubusercontent.com/clir/clearnlp-guidelines/master/md/specifications/dependency_labels.md'

    txt = urllib.request.urlopen(clear_nlp_readme_url).read().decode("utf-8")
    group_descriptions = txt.split('\n## ')[3:]

    rel_to_group = {'cop': 'Miscellaneous', 'intj': 'Miscellaneous', 'nn': 'Compound', 'nounmod': 'Nominals',
        'obj': 'Objects', 'obl': 'Miscellaneous', 'quantmod': 'Miscellaneous', 'npadvmod': 'Noun'}

    for group_desc in group_descriptions:
        clean_desc = group_desc.replace('###', '')
        desc_words = clean_desc.split()
        group_name = desc_words[0]

        rels = [w[2: -2] for w in desc_words if w.startswith("(`")]
        for rel in rels:
            rel_to_group[rel] = group_name

    rel_to_group['ROOT'] = 'ROOT'
    rel_to_group['_'] = '_'

    for filename in tqdm(glob.glob(str(DATA_DIR / 'csv') + "/*/*.csv")):
        df = pd.read_csv(filename, index_col=None)
        df['REL_GROUP'] = df.apply(lambda row: rel_to_group[row['DEP_REL']], axis=1)
        df.to_csv(filename, index=False)
        
def generate_dep_relations_txt():
    path = DATA_DIR / 'csv' 
    all_files = glob.glob(str(path) + "/*/*.csv")
    dfs = [pd.read_csv(filename, index_col=None) for filename in all_files]
    df = pd.concat(dfs, axis=0, ignore_index=False)
    dep_relations = [l.lower() for l in set(df['DEP_REL'])] 
    # Save dep_relations.txt
    with open(LIBERT_DIR / 'dep_relations.txt', 'w', encoding='utf-8') as dep_rel_labels_f:
        dep_rel_labels_f.write('\n'.join(dep_relations))

# if __name__ == "__main__":
#     parse_cross_domain(domains=['restaurants', 'laptops', 'device'])
#     # parse_in_domain(domains=['restaurants', 'laptops', 'device'])
    
#     # Save label set to label.txt
#     with open(DATA_DIR / 'csv' / 'labels.txt', 'w', encoding='utf-8') as labels_f:
#         labels_f.write('\n'.join(['O', 'B-ASP', 'I-ASP', 'B-OP', 'I-OP']))

#     # Prepare dep_relations.txt with all dependecny relation labels from ALL CSVs
#     generate_dep_relations_txt()

#     # Add column to ALL CSVs containing relation group which each relation type belongs to
#     add_rel_group_column()

add_rel_group_column()