# Most Common Word Sense

The most common word sense algorithm's goal is to extract the most common sense of a target word. the input to the algorithm is the target word and the output are the senses of the target word where each sense is scored according to the most commonly used sense in the language.
note that most of the words in the language have many senses. the sense of a word a consists of the definition of the word and the inherited hypernyms of the word.
for example: the most common sense of the target_word 'burger' is:
`definition: "a sandwich consisting of a fried cake of minced beef served on a bun, often with other ingredients"
 inherited hypernyms: ['sandwich', 'snack_food']`
 whereas the least common sense is:
 `definition: "United States jurist appointed chief justice of the United States Supreme Court by Richard Nixon (1907-1995)"`

Our approach:
Training: the training inputs a list of target_words where each word is associated with a correct (true example)
or incorrect (false example) sense. the sense consists of the definition and the inherited hypernyms of the target word in a specific sense.
Inference: extracts all the possible senses for a specific target_word and scores those senses according to the most common sense of the target_word. the higher the score the higher the probability of the sense being the most commonly used sense.

both in the training and inference a feature vector is constructed as input to the neural network. the feature vector consists of:
 - the word embedding distance between the target_word and the inherited hypernyms
 - 2 variations of the word embedding distance between the target_word and the definition
 - the word embedding of the target_word
 - the CBOW word embedding of the definition

## Requirements:
The training module inputs a gold standard csv file which is list of target_words where each word is associated with a CLASS_LABEL - a correct (true example) or an incorrect (false example) sense. the sense consists of the definition and the inherited hypernyms of the target word in a specific sense.
the user needs to prepare this gold standard csv file in advance. The file should include the following 4 columns:
| TARGET_WORD | DEFINITION | SEMANTIC_BRANCH | CLASS_LABEL
where:
1. TARGET_WORD(string):the word that you want to get the most common sense of. e.g. chair
2. DEFINITION (string): the definition of the word (usually a single sentence) extracted from external resource such as wordnet or wikidata. e.g. an articat that is design for sitting
3. SEMANTIC_BRANCH(string):  [comma seoarated] the inherited hypernyms extracted from external resource such as wordnet or wikidata e.g. [funniture, articact]
4. CLASS_LABEL(string): a binary Y value 0/1 that represent whether the sense (Definition and semantic branch) is the most common sense  of the target word. e.g. 1

Store the file in the data folder of the project.
## Dependencies:
- python version 3.6.3 was used in this project
- **nervananeon** (version 2.4.0 is used in this project)
- **aeon** (version 2.0.2 is used in this project)
- **gensim** (version 3.2.0. in prepare_data.py, used for basic word embedding's utilities)
- **nltk** (version 3.2.5 in Inference.py, used for extracting all senses of each word)
- **sklearn** (version 0.19.0. in prepare_data.py, used for splitting train and validation sets)
- **numpy**
- **csv** (in prepare_data.py, used for reading gold standard files)
- **codecs**
- **math**
- **pickle** (used for saving and reading data files)
- **termcolor** (in Inference.py, used for highlighting the detected sense)

## Prepare training and validation test sets

`python prepare_data [--gold_standard_file GOLD_STANDARD_FILE]
       [--word_embedding_model_file WORD_EMBEDDING_MODEL_FILE]
       [--training_to_validation_size_ratio RATIO_BETWEEN_TRAINNIG_AND_VALIDATIO]
       [--data_set_file DATA_SET_FILE]`

### Example:

1. preparing data for training and validation using pre-trained Google News Word2vec model.
The terms and conditions of the data set license apply. Intel does not grant any rights to the data files or database.
see relevant license agreement http://www.apache.org/licenses/LICENSE-2.0
Pretrained Google News Word2vec model can be download at <https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing>.
2. Prepare in advance gold standard csv file as described in the requirements section above.

`python prepare_data.py --gold_standard_file data/gold_standard.csv
       --word_embedding_model_file pre-trained_models/GoogleNews-vectors-negative300.bin
       --training_to_validation_size_ratio 0.8
       --data_set_file data/data_set.pkl`

## Training
Train the MLP classifier and evaluate it.

`python train.py [--data_set_file DATA_SET_FILE] [--model_prm MODEL_PRM]`

### Example:
`python train.py --data_set_file data/data_set.pkl
                 --model_prm data/wsd_classification_model.prm`

## Inference
`python inference.py [--max_num_of_senses_to_search N]
        [--input_inference_examples_file INPUT_INFERENCE_EXAMPLE_FILE]
        [--model_prm MODEL_PRM] [--word_embedding_model_file WORD_EMBEDDING_MODEL_FILE]`

### Example:

`python inference.py --max_num_of_senses_to_search 3
       --input_inference_examples_file data/input_inference_examples.csv
       --word_embedding_model_file pretrained_models/GoogleNews-vectors-negative300.bin
       --model_prm data/wsd_classification_model.prm`

where:
max_num_of_senses_to_search is the maximum number of senses that are checked per target word . default =3
input_inference_examples_file is a csv file containing the input inference data. this file includes a single column wherein each entry in this column is a different target word (see input_inference_examples.csv under the data folder):

note that the results are printed to the terminal using different colors therefore using a white terminal background is best to view the results
