import pickle

from neon import logger as neon_logger
from neon.backends import gen_backend
from neon.callbacks.callbacks import Callbacks
from neon.layers import GeneralizedCost
from neon.optimizers.optimizer import RMSProp
from neon.transforms.cost import CrossEntropyMulti
from neon.util.argparser import NeonArgparser, extract_valid_args

from data import CONLL2000
from model import ChunkerModel
from utils import label_precision_recall_f1

if __name__ == '__main__':

    parser = NeonArgparser()
    parser.add_argument('--use_w2v', default=False, action='store_true',
                        help='Use pre-trained word embedding from given w2v model path')
    parser.add_argument('--embedding_model', type=str,
                        help='w2v embedding model path (only GloVe and Fasttext are supported')
    parser.add_argument('--use_pos', default=False, action='store_true',
                        help='Use part-of-speech tags of tokens')
    parser.add_argument('--use_char_rnn', default=False, action='store_true',
                        help='Use char-RNN features of tokens')
    parser.add_argument('--sentence_len', default=100, type=int,
                        help='Sentence token length')
    parser.add_argument('--lstm_depth', default=1, type=int,
                        help='Deep BiLSTM depth')
    parser.add_argument('--lstm_hidden_size', default=100, type=int,
                        help='LSTM cell hidden vector size')
    parser.add_argument('--token_embedding_size', default=50, type=int,
                        help='Token embedding vector size')
    parser.add_argument('--pos_embedding_size', default=25, type=int,
                        help='Part-of-speech embedding vector size')
    parser.add_argument('--vocab_size', default=25000, type=int,
                        help='Vocabulary size to use (only if pre-trained embedding is not used)')
    parser.add_argument('--char_hidden_size', default=25, type=int,
                        help='Char-RNN cell hidden vector size')
    parser.add_argument('--max_char_word_length', default=20, type=int,
                        help='max characters per one word')
    parser.add_argument('--model_name', default='chunker', type=str,
                        help='Model file name')
    parser.add_argument('--settings', default='chunker_settings', type=str,
                        help='Model settings file name')
    parser.add_argument('--print_np_perf', default=True, action='store_true',
                        help='Print Noun Phrase (NP) tags accuracy')

    args = parser.parse_args(gen_be=False)
    if args.use_pos:
        pos_vocab_size = 50
    else:
        pos_vocab_size = None
    if args.use_char_rnn:
        char_vocab_size = 82
    else:
        char_vocab_size = None
    be = gen_backend(**extract_valid_args(args, gen_backend))

    dataset = CONLL2000(sentence_length=args.sentence_len,
                        vocab_size=args.vocab_size,
                        use_pos=args.use_pos,
                        use_chars=args.use_char_rnn,
                        chars_len=args.max_char_word_length,
                        use_w2v=args.use_w2v,
                        w2v_path=args.embedding_model)
    train_set = dataset.train_iter
    test_set = dataset.test_iter

    model = ChunkerModel(sentence_length=args.sentence_len,
                         token_vocab_size=args.vocab_size,
                         pos_vocab_size=pos_vocab_size,
                         char_vocab_size=char_vocab_size,
                         max_char_word_length=args.max_char_word_length,
                         token_embedding_size=args.token_embedding_size,
                         pos_embedding_size=args.pos_embedding_size,
                         char_embedding_size=args.char_hidden_size,
                         num_labels=dataset.y_size,
                         lstm_hidden_size=args.lstm_hidden_size,
                         num_lstm_layers=args.lstm_depth,
                         embedding_model=args.embedding_model)

    cost = GeneralizedCost(costfunc=CrossEntropyMulti(usebits=True))
    optimizer = RMSProp(stochastic_round=args.rounding)
    callbacks = Callbacks(model.get_model(), eval_set=test_set, **args.callback_args)
    model.fit(train_set,
              optimizer=optimizer,
              epochs=args.epochs,
              cost=cost,
              callbacks=callbacks)

    # save model
    model_settings = {'sentence_len': args.sentence_len,
                      'use_embeddings': args.use_w2v,
                      'pos': args.use_pos,
                      'char_rnn': args.use_char_rnn,
                      'y_vocab': dataset.y_vocab
                      }

    model_settings.update({'vocabs': dataset.vocabs})
    if args.use_w2v is True:
        model_settings.update({'embedding_size': dataset.emb_size})

    with open(args.settings + '.dat', 'wb') as fp:
        pickle.dump(model_settings, fp)
    model.save('{}.prm'.format(args.model_name))

    # tagging accuracy
    y_preds = model.predict(test_set)
    shape = (test_set.nbatches, args.batch_size, args.sentence_len)
    prediction = y_preds.argmax(2).reshape(shape).transpose(1, 0, 2)
    fraction_correct = (prediction == test_set.y).mean()
    neon_logger.display('Misclassification error = %.1f%%' % ((1 - fraction_correct) * 100))

    # check NP label accuracy
    if args.print_np_perf is True:
        np_labels = [dataset.y_vocab['B-NP'] + 1, dataset.y_vocab['I-NP'] + 1]
        p, r, f1 = label_precision_recall_f1(test_set.y.reshape(-1, args.sentence_len),
                                             prediction.reshape(-1, args.sentence_len),
                                             np_labels)
        print(p, r, f1)
