from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import re

import ftfy
import textacy
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from textacy.preprocess import unpack_contractions
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES

LINE_BREAK = re.compile(r'(\r\n)+')
SPACE_CHAR = re.compile(r'(?!\n)\s+')
PARAN_CHARS = re.compile(r'[\(\)<>{}]')
SPECIAL_CHARS = re.compile(r'["*!`~;:?/@#%]')


def fix_whitespaces(text):
    return SPACE_CHAR.sub(' ', LINE_BREAK.sub(r'\n', text)).strip()


def fix_encoding(text):
    return ftfy.fix_text(text)


def remove_urls(text):
    return textacy.preprocess.replace_urls(text, '')


def remove_emails(text):
    return textacy.preprocess.replace_emails(text, '')


def remove_phones(text):
    return textacy.preprocess.replace_phone_numbers(text, '')


def remove_accents(text):
    return textacy.preprocess.remove_accents(text)


def remove_parentheses(text):
    return PARAN_CHARS.sub(' ', text)


def remove_special(text):
    return SPECIAL_CHARS.sub('', text)


def remove_non_alphanumeric(text):
    return filter(str.isalnum, text)


class TextCleaner:
    """
    Text cleaning module. Supports arbitrary number of
    str->str cleaning methods.
    Arguments:
        rules(list): a list of cleaning methods.
    """

    DEFAULT_RULES = [
        fix_encoding,
        remove_accents,
        remove_urls,
        remove_emails,
        remove_phones,
        unpack_contractions,
        remove_parentheses,
        remove_special,
        fix_whitespaces,
    ]

    def __init__(self, rules=DEFAULT_RULES):
        self.rules = rules

    def process(self, text):
        """
        Process text through all given rules
        """
        n_text = text
        while True:
            last_text = n_text
            for r in self.rules:
                n_text = r(n_text)
            if last_text == n_text:
                break
        return n_text

    def process_list(self, texts):
        return [self.process(t) for t in texts]


stemmer = EnglishStemmer()
lemmatizer = WordNetLemmatizer()
p = re.compile(r'[ \-,;.@&]')


def simple_normalizer(text):
    """
    Simple text normalizer. Runs each token of a phrase thru a lemmatizer
    and a stemmer.
    """
    if not str(text).isupper() or \
            not str(text).endswith('S') or \
            not len(text.split()) == 1:
        tokens = list(filter(lambda x: len(x) != 0, p.split(text.strip())))
        text = ' '.join([stemmer.stem(lemmatizer.lemmatize(t))
                         for t in tokens])
    return text

spacy_lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)

def spacy_normalizer(text, lemma=None):
    """
    Simple text normalizer using spacy lemmatizer. Runs each token of a phrase
    thru a lemmatizer and a stemmer.
    Arguments:
        text(string): the text to normalize.
        lemma(string): lemma of the given text. in this case only stemmer will
        run.
    """
    if not str(text).isupper() or \
            not str(text).endswith('S') or \
            not len(text.split()) == 1:
        tokens = list(filter(lambda x: len(x) != 0, p.split(text.strip())))
        if lemma:
            lemma = lemma.split(' ')
            text = ' '.join([stemmer.stem(l)
                             for l in lemma])
        else:
            text = ' '.join([stemmer.stem(spacy_lemmatizer(t, u'NOUN')[0])
                             for t in tokens])
    return text

class Stopwords:
    """
    Stopwords class. loads a dictionary (english) and enables checking if
    given word is a stop word.
    """
    instance = None

    class SW:
        def __init__(self):
            self.sw = set(stopwords.words('english'))

    def __init__(self):
        if not Stopwords.instance:
            Stopwords.instance = Stopwords.SW()

    def is_stopword(self, word):
        return word in self.instance.sw


if __name__ == '__main__':
    # text = 'this is a sample    text with ßåµπ¬'
    sample_text = 'this is a sample    ' \
                  'http://www.google.com/dbla?sdfjowef=jfsdf&gjosi' \
           'djf&fgsdf=3 text with email@google.com and my phone ' \
           'number is +972123123123        ' \
                  'sjdf isdof sijf < > s<<<<  () sdf sdSD F) s( S)) .' \
                  'isn\'t this great?'
    p = TextCleaner()
    c_text = p.process(sample_text)
    print('Before:\t\t{}'.format(sample_text))
    print('After:\t\t{}'.format(c_text))