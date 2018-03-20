from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
import argparse
import os

from keras.callbacks import ModelCheckpoint

from data import ATIS, SNIPS
from model import JointSequentialLSTM
from utils import ConllCallback, get_conll_scores

parser = argparse.ArgumentParser()
parser.add_argument('-b', type=int, default=10,
                    help='Batch size')
parser.add_argument('-e', type=int, default=10,
                    help='Number of epochs')
parser.add_argument('--dataset', type=str, required=True,
                    choices=['atis', 'snips'],
                    help='Dataset for training the model')
parser.add_argument('--sentence_length', type=int, default=30,
                    help='Max sentence length')
parser.add_argument('--token_emb_size', type=int, default=100,
                    help='Token features embedding vector size')
parser.add_argument('--intent_hidden_size', type=int, default=100,
                    help='Intent detection LSTM hidden size (joint model only)')
parser.add_argument('--lstm_hidden_size', type=int, default=150,
                    help='Slot tags LSTM hidden size (joint model only)')
parser.add_argument('--tagger_dropout', type=float, default=0.5,
                    help='Slot tags dropout value (joint model only)')
parser.add_argument('--embedding_path', type=str,
                    help='Path to word embedding model file')
parser.add_argument('--full_eval', action='store_true', default=False,
                    help='Print full slot tagging statistics instead of global P/R/F1')
parser.add_argument('--restore', action='store_true', default=False,
                    help='Restore model weights from model path')
parser.add_argument('--model_path', type=str, default='model.h5',
                    help='Model file path')
parser.add_argument('--save_epochs', type=int, default=1,
                    help='Number of epochs to run between model saves')
args = parser.parse_args()


# load dataset
if args.dataset == 'atis':
    dataset = ATIS(sentence_length=args.sentence_length,
                   embedding_model=args.embedding_path,
                   embedding_size=args.token_emb_size)
if args.dataset == 'snips':
    dataset = SNIPS(sentence_length=args.sentence_length,
                    embedding_model=args.embedding_path,
                    embedding_size=args.token_emb_size)

train_x, train_i, train_y = dataset.train_set
test_x, test_i, test_y = dataset.test_set

model = JointSequentialLSTM()

if args.restore and os.path.exists(args.model_path):
    print('Loading model weights and continuing with training ..')
    model.load(args.model_path)
else:
    print('Creating new model, starting to train from scratch')
    model.build(args.sentence_length,
                dataset.vocab_size,
                dataset.slot_vocab_size,
                dataset.intent_size,
                args.token_emb_size,
                args.lstm_hidden_size,
                args.tagger_dropout,
                args.intent_hidden_size,
                args.embedding_path)

conll_cb = ConllCallback(test_x, test_y, dataset.slots_vocab, batch_size=args.b)
cp_cb = ModelCheckpoint(args.model_path, verbose=1, period=args.save_epochs)

# train model
model.fit(x=train_x, y=[train_i, train_y],
          batch_size=args.b, epochs=args.e,
          validation=(test_x, [test_i, test_y]),
          callbacks=[conll_cb, cp_cb])
print('Training done.')

# test performance
predictions = model.predict(test_x, batch_size=args.b)
eval = get_conll_scores(predictions, test_y, {
                        v: k for k, v in dataset.slots_vocab.items()})
if args.full_eval is True:
    print(eval)
else:
    print(eval[0])
