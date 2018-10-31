# Most Common Word Sense
The most common word sense algorithm's goal is to extract the most common sense of a target word. The input to the algorithm is the target word and the output are the senses of the target word where each sense is scored according to the most commonly used sense in the language.

## Prepare training and validation test sets
The training module inputs a gold standard csv file which is list of target_words where each word is associated with a CLASS_LABEL - a correct (true example) or an incorrect (false example) sense. the sense consists of the definition and the inherited hypernyms of the target word in a specific sense.
The user needs to prepare this gold standard csv file in advance. The file should include the following 4 columns:

| TARGET_WORD | DEFINITION | SEMANTIC_BRANCH | CLASS_LABEL

where:
1. TARGET_WORD(string):the word that you want to get the most common sense of. e.g. chair
2. DEFINITION (string): the definition of the word (usually a single sentence) extracted from external resource such as Wordnet or Wikidata. e.g. an artifact that is design for sitting
3. SEMANTIC_BRANCH(string):  [comma separated] the inherited hypernyms extracted from external resource such as Wordnet or Wikidata e.g. [furniture, artifact]
4. CLASS_LABEL(string): a binary value 0/1 that represents whether the sense (Definition and semantic branch) is the most common sense of the target word.

`python prepare_data [--gold_standard_file GOLD_STANDARD_FILE]
       [--word_embedding_model_file WORD_EMBEDDING_MODEL_FILE]
       [--training_to_validation_size_ratio RATIO_BETWEEN_TRAINNIG_AND_VALIDATIO]
       [--data_set_file DATA_SET_FILE]`

## Training
Train the MLP classifier and evaluate it.

`python train.py [--data_set_file DATA_SET_FILE] [--model MODEL_h5]`

Quick example:

`python train.py --data_set_file data/data_set.pkl
                 --model data/wsd_classification_model.h5`

## Inference
When running inference note that the results are printed to the terminal using different colors therefore using a white terminal background is best to view the results.

`python inference.py [--max_num_of_senses_to_search N]
        [--input_inference_examples_file INPUT_INFERENCE_EXAMPLE_FILE]
        [--model MODEL_H5] [--word_embedding_model_file WORD_EMBEDDING_MODEL_FILE]`

Quick example:

`python inference.py --max_num_of_senses_to_search 3
       --input_inference_examples_file data/input_inference_examples.csv
       --word_embedding_model_file pretrained_models/GoogleNews-vectors-negative300.bin
       --model data/wsd_classification_model.h5`
