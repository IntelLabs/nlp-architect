
from nltk.stem.snowball import EnglishStemmer
from nltk import WordNetLemmatizer
import re



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