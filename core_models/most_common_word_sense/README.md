# Most Common Word Sense
The most common word sense algorithm's goal is to extract the most common sense of a target word. The input to the algorithm is the target word and the output are the senses of the target word where each sense is scored according to the most commonly used sense in the language.

## Prepare training and validation test sets

`python prepare_data [--gold_standard_file GOLD_STANDARD_FILE]
       [--word_embedding_model_file WORD_EMBEDDING_MODEL_FILE]
       [--training_to_validation_size_ratio RATIO_BETWEEN_TRAINNIG_AND_VALIDATIO]
       [--data_set_file DATA_SET_FILE]`

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

