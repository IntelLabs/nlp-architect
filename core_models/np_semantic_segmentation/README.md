# Noun Phrase (NP) Semantic Segmentation

Noun-Phrase (NP) is a phrase which has a noun (or pronoun) as its head and zero of more dependent modifiers.
Noun-Phrase is the most frequently occurring phrase type and its inner segmentation is critical for understanding the
semantics of the Noun-Phrase.
The most basic division of the semantic segmentation is to two classes:
1. Descriptive Structure - a structure where all dependent modifiers are not changing the semantic meaning of the Head.
2. Collocation Structure - a sequence of words or term that co-occur and change the semantic meaning of the Head.

For example:
- `fresh hot dog` - hot dog is a collocation, and changes the head (`dog`) semantic meaning.
- `fresh hot pizza` - fresh and hot are descriptions for the pizza.

This model is the first step in the Semantic Segmentation algorithm - the MLP classifier.
The Semantic Segmentation algorithm takes the dependency relations between the Noun-Phrase words, and the MLP classifier inference as the
input - and build a semantic hierarchy that represents the semantic meaning.
The Semantic Segmentation algorithm eventually create a tree where each tier represent a semantic meaning -> if a sequence of words is a
collocation then a collocation tier is created, else the elements are broken down and each one is mapped
to different tier in the tree.

This model trains MLP classifier and inference from such classifier in order to conclude the correct segmentation
for the given NP.

for the examples above the classifier will output 1 (==Collocation) for `hot dog` and output 0 (== not collocation)
for `hot pizza`.


## Requirements:
- **neon**
- **nltk** (for data.py - used for Wordnet, SnowballStemmer)
- **palmettopy** (for data.py - used for Palmetto PMI scores)
- **requests** (for data.py - used for Wikidata)
- **gensim** (for data.py - used for Word2Vec utilities)
- **tqdm** (for data.py)
- **numpy**

## Files
- *preprocess_tratz2011.py*: Constructing labeled dataset from Tratz 2011 dataset.
- *data.py*: Prepare string data for both `train.py` and `inference.py` using pre-trained word embedding, PMI score, Wordnet and wikidata.
- *feature_extraction.py*: contains the feature extraction services
- *train.py*: train the MLP classifier.
- *model.py*: contains the MLP classifier model.
- *inference.py*: load the trained model and inference the input data by the model.

## Dataset
The expected dataset is a CSV file with 2 columns. the first column contains the Noun-Phrase string (a Noun-Phrase containing 2 words), and the second column contains the correct label (if the 2 word Noun-Phrase is a collocation - the label is 1, else 0)

If you wish to use an existing dataset for training the model, you can download Tratz 2011 et al. dataset [1,2] from the following link:
[Tratz 2011 Dataset](https://vered1986.github.io/papers/Tratz2011_Dataset.tar.gz). Is also available in [here](https://www.isi.edu/publications/licensed-sw/fanseparser/index.html).
(The terms and conditions of the data set license apply. Intel does not grant any rights to the data files or database. see relevant [license agreement](http://www.apache.org/licenses/LICENSE-2.0))


After downloading and unzipping the dataset, run `preprocess_tratz2011.py` in order to construct the labeled data and save it in a CSV file (as expected for the model).
the scripts read 2 .tsv files ('tratz2011_coarse_grained_random/train.tsv' and 'tratz2011_coarse_grained_random/val.tsv') and outputs 2 .csv files accordingly.

Parameters can be obtained by running:

    python preprocess_tratz2011.py -h
        --data path_to_Tratz_2011_dataset_folder


### Pre-processing the data:
A feature vector is extracted from each Noun-Phrase string:

* Word2Vec word embedding (300 size vector for each word in the Noun-Phrase) .
    * Pre-trained Google News Word2vec model can download [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)
    * The terms and conditions of the data set license apply. Intel does not grant any rights to the data files or database. see relevant [license agreement](http://www.apache.org/licenses/LICENSE-2.0)
* Cosine distance between 2 words in the Noun-Phrase.
* PMI score (NPMI and UCI scores).
* A binary features whether the Noun-Phrase has existing entity in Wikidata.
* A binary features whether the Noun-Phrase has existing entity in WordNet.

#### pre-processing the dataset:
Parameters can be obtained by running:

    python data.py -h
        --data DATA           path the CSV file where the raw dataset is saved
        --output OUTPUT       path the CSV file where the prepared dataset will be saved
        --w2v_path W2V_PATH   path to the word embedding's model (default: None)
        --http_proxy HTTP_PROXY     system's http proxy (default: None)
        --https_proxy HTTPS_PROXY   system's https proxy (default: None)

Quick example:

    python data.py --data input_data_path.csv --output output_prepared_path.csv --w2v_path <path_to_w2v>/GoogleNews-vectors-negative300.bin.gz

## Training:
Train the MLP classifier and evaluate it.
Parameters can be obtained by running:

    python train.py -h
        --data DATA           Path to the CSV file where the prepared dataset is saved
        --model_path MODEL_PATH     Path to save the model

After training is done, the model is saved automatically:

`<model_name>.prm` - the trained model

Quick example:

    python train.py --data prepared_data_path.csv --model np_semantic_segmentation_path.prm

## Inference:
In order to run inference you need to have pre-trained `<model_name>.prm` file and data CSV file
that was generated by `prepare_data.py`.
The result of `inference.py` is a CSV file, each row contains the model's inference in respect to the input data.

    python inference.py -h
        --data DATA           prepared data CSV file path (default: None)
        --model MODEL         path to the trained model file (default: None)
        --print_stats PRINT_STATS       print evaluation stats for the model predictions - if your data has tagging (default: False)
        --output OUTPUT       path to location for inference output file (default: None)
Quick example:

    python inference.py --model np_semantic_segmentation_path.prm --data prepared_data_path.csv --output inference_data.csv --print_stats True


## Citations:
[1] Tratz, Stephen, and Eduard Hovy. ["A taxonomy, dataset, and classifier for automatic noun compound interpretation."] (http://www.aclweb.org/anthology/P10-1070) Proceedings of the 48th Annual Meeting of the Association for Computational Linguistics. Association for Computational Linguistics, 2010.
[2] Shwartz, Vered, and Chris Waterson. ["Olive Oil is Made of Olives, Baby Oil is Made for Babies: Interpreting Noun Compounds using Paraphrases in a Neural Model."] (https://arxiv.org/pdf/1803.08073.pdf) arXiv preprint arXiv:1803.08073 (2018).