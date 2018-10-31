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

We use the [CONLL2000](https://www.clips.uantwerpen.be/conll2000/chunking/) shared task dataset 
in the default parameters of our example for training np2vec model. The terms and conditions 
of the data set license apply. Intel does not grant any rights to the data files.

## Training Usage

```
usage: train.py [-h] [--corpus CORPUS] [--corpus_format {json,txt,conll200}]
                     [--mark_char MARK_CHAR]
                     [--word_embedding_type {word2vec,fasttext}]
                     [--np2vec_model_file NP2VEC_MODEL_FILE] [--binary]
                     [--sg {0,1}] [--size SIZE] [--window WINDOW]
                     [--alpha ALPHA] [--min_alpha MIN_ALPHA]
                     [--min_count MIN_COUNT] [--sample SAMPLE]
                     [--workers WORKERS] [--hs {0,1}] [--negative NEGATIVE]
                     [--cbow_mean {0,1}] [--iter ITER] [--min_n MIN_N]
                     [--max_n MAX_N] [--word_ngrams {0,1}]

optional arguments:
  -h, --help            show this help message and exit
  --corpus CORPUS       path to the corpus. By default, it is the training set of CONLL2000 
                        shared task dataset.
  --corpus_format {json,txt,conll2000}
                        format of the input marked corpus; txt, json and conll2000
                        formats are supported. For json format, the file
                        should contain an iterable of sentences. Each sentence
                        is a list of terms (unicode strings) that will be used
                        for training. (Default: conll2000)
  --mark_char MARK_CHAR
                        special character that marks word separator and NP suffix.
  --word_embedding_type {word2vec,fasttext}
                        word embedding model type; word2vec and fasttext are
                        supported.
  --np2vec_model_file NP2VEC_MODEL_FILE
                        path to the file where the trained np2vec model has to
                        be stored. (Default: conll2000.train.model)
  --binary              boolean indicating whether the model is stored in
                        binary format; if word_embedding_type is fasttext and
                        word_ngrams is 1, binary should be set to True.
  --sg {0,1}            model training hyperparameter, skip-gram. Defines the
                        training algorithm. If 1, CBOW is used, otherwise,
                        skip-gram is employed.
  --size SIZE           model training hyperparameter, size of the feature
                        vectors.
  --window WINDOW       model training hyperparameter, maximum distance
                        between the current and predicted word within a
                        sentence.
  --alpha ALPHA         model training hyperparameter. The initial learning
                        rate.
  --min_alpha MIN_ALPHA
                        model training hyperparameter. Learning rate will
                        linearly drop to `min_alpha` as training progresses.
  --min_count MIN_COUNT
                        model training hyperparameter, ignore all words with
                        total frequency lower than this.
  --sample SAMPLE       model training hyperparameter, threshold for
                        configuring which higher-frequency words are randomly
                        downsampled, useful range is (0, 1e-5)
  --workers WORKERS     model training hyperparameter, number of worker
                        threads.
  --hs {0,1}            model training hyperparameter, hierarchical softmax.
                        If set to 1, hierarchical softmax will be used for
                        model training. If set to 0, and `negative` is non-
                        zero, negative sampling will be used.
  --negative NEGATIVE   model training hyperparameter, negative sampling. If >
                        0, negative sampling will be used, the int for
                        negative specifies how many "noise words" should be
                        drawn (usually between 5-20). If set to 0, no negative
                        sampling is used.
  --cbow_mean {0,1}     model training hyperparameter. If 0, use the sum of
                        the context word vectors. If 1, use the mean, only
                        applies when cbow is used.
  --iter ITER           model training hyperparameter, number of iterations.
  --min_n MIN_N         fasttext training hyperparameter. Min length of char
                        ngrams to be used for training word representations.
  --max_n MAX_N         fasttext training hyperparameter. Max length of char
                        ngrams to be used for training word representations.
                        Set `max_n` to be lesser than `min_n` to avoid char
                        ngrams being used.
  --word_ngrams {0,1}   fasttext training hyperparameter. If 1, uses enrich
                        word vectors with subword (ngrams) information. If 0,
                        this is equivalent to word2vec training.
```

## Inference Usage

```
usage: inference.py [-h] [--np2vec_model_file NP2VEC_MODEL_FILE]
                         [--binary] [--word_ngrams {0,1}]

optional arguments:
  -h, --help            show this help message and exit
  --np2vec_model_file NP2VEC_MODEL_FILE
                        path to the file with the np2vec model to load. (Default: conll2000.train.model)
  --binary              boolean indicating whether the model to load has been
                        stored in binary format.
  --word_ngrams {0,1}   If 0, the model to load stores word information. If 1,
                        the model to load stores subword (ngrams) information;
                        note that subword information is relevant only to
                        fasttext models.
  --mark_char MARK_CHAR
                        special character that marks word separator and NP suffix.
  --np NP               NP to print its word vector.
```
                        
More details about the hyperparameters at <https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec> for word2vec and <https://radimrehurek.com/gensim/models/fasttext.html#gensim.models.fasttext.FastText> for fasttext. 

