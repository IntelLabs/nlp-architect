
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
from nltk.stem.snowball import EnglishStemmer
from nltk import WordNetLemmatizer
import re


stemmer = EnglishStemmer()
nltk_lemmatizer = WordNetLemmatizer()
lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)


p = re.compile(r'[ \-,;.@&]')


def spacy_normalizer(text, lemma=None):
    """
    Simple text normalizer. Runs each token of a phrase thru a lemmatizer
    and a stemmer.
    Arg: lemma: providing a lemma string. in this case only stemmer will run.
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
            text = ' '.join([stemmer.stem(lemmatizer(t, u'NOUN')[0])
                             for t in tokens])
    return text


def nltk_normalizer(text):
    """
    Simple text normalizer. Runs each token of a phrase thru a lemmatizer
    and a stemmer.
    """
    if not str(text).isupper() or \
            not str(text).endswith('S') or \
            not len(text.split()) == 1:
        tokens = list(filter(lambda x: len(x) != 0, p.split(text.strip())))
        text = ' '.join([stemmer.stem(nltk_lemmatizer.lemmatize(t))
                         for t in tokens])
    return text