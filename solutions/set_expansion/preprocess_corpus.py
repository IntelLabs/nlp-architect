import logging
import sys
import spacy
from configargparse import ArgumentParser

from nlp_architect.utils.io import check_size

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def mark_np(np):
    return np.replace(' ', '_') + '_'


if __name__ == "__main__":
    arg_parser = ArgumentParser(__doc__)
    arg_parser.add_argument(
        '--corpus',
        default='train.txt',
        type=str,
        action=check_size(min=1),
        help='path to the input corpus. By default, '
             'it is the training set of CONLL2000 shared task dataset.')
    arg_parser.add_argument(
        '--marked_corpus',
        default='marked_train.txt',
        type=str,
        action=check_size(min=1),
        help='path to the corpus. By default, '
             'it is the training set of CONLL2000 shared task dataset.')

    arg_parser.add_argument(
        '--mark_char',
        default='_',
        type=str,
        action=check_size(1, 2),
        help='special character that marks word separator and NP suffix.')

    args = arg_parser.parse_args()

    logger.info('loading spacy')
    nlp = spacy.load('en_core_web_sm', disable=['textcat', 'ner'])
    logger.info('spacy loaded')

    corpus_file = open(args.corpus, 'r', encoding='utf8')
    marked_corpus_file = open(args.marked_corpus, 'w', encoding='utf8')

    num_lines = sum(1 for line in corpus_file)
    corpus_file.seek(0)
    logger.info(str(num_lines) + ' lines in corpus')
    i = 0

    for doc in nlp.pipe(corpus_file):
        spans = list()
        for p in doc.noun_chunks:
            spans.append(p)
        i += 1
        if len(spans) > 0:
            span = spans.pop(0)
        else:
            span = None
        spanWritten = False
        for token in doc:
            if span is None:
                if len(token.text.strip()) > 0:
                    marked_corpus_file.write(token.text + ' ')
            else:
                if token.idx < span.start_char or token.idx >= span.end_char:  # outside a span
                    if len(token.text.strip()) > 0:
                        marked_corpus_file.write(token.text + ' ')
                else:
                    if not spanWritten:
                        text = span.text.replace(' ', args.mark_char) + args.mark_char
                        marked_corpus_file.write(text + ' ')
                        spanWritten = True
                    if token.idx + len(token.text) == span.end_char:
                        if len(spans) > 0:
                            span = spans.pop(0)
                        else:
                            span = None
                        spanWritten = False
        marked_corpus_file.write('\n')
        if i % 1000 == 0:
            logger.info(str(i) + ' of ' + str(num_lines) + ' lines')

    marked_corpus_file.flush()
    marked_corpus_file.close()
