# NP2vec - Word Embedding's model training for Noun Phrases

Noun Phrases (NP) play a particular role in NLP algorithms.
This code consists in training a word embedding's model for Noun NP's using [word2vec](https://code.google.com/archive/p/word2vec/) or [fasttext](https://github.com/facebookresearch/fastText) algorithm.
It assumes that the NP's are already extracted and marked in the input corpus.
All the terms in the corpus are used as context in order to train the word embedding's model; however, at the end of the training, only the word embedding's of the NP's are stored, except for the case of
fasttext training with word_ngrams=1; in this case, we store all the word embedding's, including non-NP's in order to be able to estimate word embeddings of out-of-vocabulary NP's (NP's that don't appear in
the training corpora).

Note that this code can be also used to train a word embedding's model on any marked corpus. For example, if you mark verbs in your corpus, you can train a verb2vec model.

NP's have to be marked in the corpus by a marking character between the words of the NP and as a suffix of the NP.
For example, if the marking character is '\_', the NP "Natural Language Processing" will be marked as "Natural_Language_Processing_".

## Training:
To train use the following command:

```
python main_train.py [-h] [--corpus CORPUS] [--corpus_format {json,txt}]
                     [--mark_char MARK_CHAR]
                     [--word_embedding_type {word2vec,fasttext}]
                     [--np2vec_model_file NP2VEC_MODEL_FILE] [--binary]
                     [--sg {0,1}] [--size SIZE] [--window WINDOW]
                     [--alpha ALPHA] [--min_alpha MIN_ALPHA]
                     [--min_count MIN_COUNT] [--sample SAMPLE]
                     [--workers WORKERS] [--hs {0,1}] [--negative NEGATIVE]
                     [--cbow_mean {0,1}] [--iter ITER] [--min_n MIN_N]
                     [--max_n MAX_N] [--word_ngrams {0,1}]
```

## Inference:
Inference on a model can then be completed using:
```
usage: main_inference.py [-h] [--np2vec_model_file NP2VEC_MODEL_FILE]
                         [--binary] [--word_ngrams {0,1}]
```
