# Application Overview

## General Usage:
The flow of our system consists of two main parts, the offline collateral generation and the application itself. For the offline collateral generation, we take a pretrained or finetuned model and process a subset of data with it, extracting information such as targets, predictions, relevant hidden states, and attention matrices to be later loaded in the app. Additionally, during this offline stage, the saved hidden states are further processed using t-SNE, before being saved. After this information has been extracted and processed, it is saved to disk.

At runtime, for each model being analyzed, these two aforementioned files must be provided. For example, if we wanted to compare between pretrained and finetuned BERT on a specific task, we would have to provide four files. The file pairs for each model are deliberately made to be independent of each other, so that the user can either run the app to analyze a single model or to compare two different models. In this latter case, we link the collateral files for the two models at runtime.

## Usage Recommendation:
If you want to use **InterpreT** for your own task, the easiest way to get it running is to create your DataFrame such that the columns match those in the provided example DataFrame for ABSA at "example_dfs/absa_df.dat" (same columns) and to create the attention dictionary using the below specifications. Then you will be able to either use or copy the `ABSATask` class in `tasks.py` to run **InterpreT** with your data. See the below sections for more details.

## System Architecture
This section will provide a brief overview of how the software components of **InterpreT** work, with the intent of helping users understand how they can modify InterpreT for their own tasks.
There are 4 main software components:
- Collateral Generation (specification provided below, no code provided)
    - This component is closely tied to a given workflow. Given an inference workflow, to create collateral requires saving and processing quantities such as attentions, hidden states, predictions, labels, etc. 
- Application (`appConfiguration.py`, `appLayout.py`, `main.py`)
    - These files start the Dash app and handle everything related to it (UI/layout, functionality, etc.).
- Tasks (`tasks.py`)
    - Central to **InterpreT** is the concept of a Task, which is an object that: 
        - Stores information related to a specific task (e.g. labels for UI elements, certain columns to keep track of in the DataFrame for plotting, etc.).
        - Holds all the collateral for plotting and visualizations.
        - Holds mappings used to associate data with their corresponding models, and also map internally used quantities to externally visible (in the Dash app UI) quantities.
        - Is created upon application startup, and then is used by the application to obtain the above data/information/mappings.
- Plotting Utilities (`plotFunc.py`)

# System, Task, and Collateral Generation Details

## How To Generate Collateral
- Create the DataFrame (see "examples_df/absa_df" for an easy example):
    - The DataFrame should have the following columns at minimum:
        - **id** (should be unique for each example, not necessarily row, since one example could have multiple tokens or sets of tokens we want to track)
        - **token** (can be another name, e.g. "aspect" or "span_token", and these values will be tracked/displayed in the t-SNE plot)
        - **sentence** (the sentence that the token belongs to)
        - **layer_*_tsne_x**  (where * should be 00, 01, ..., 99 depending on the number of layers in the model, one column per layer)
        - **layer_*_tsne_y**  (where * should be 00, 01, ..., 99 depending on the number of layers in the model, one column per layer)
    - To get the t-SNE coordinates, you will need to keep track of the hidden states (for each layer) for all of the tokens you are interested in, and perform t-SNE on these lists of hidden states.
        - We perform t-SNE separately for each layer, i.e. we have **num_layers + 1** many lists (+1 for the pre-Transformer embeddings) of hidden states and perform t-SNE separately on each of these layers. Note that any dimensionality reduction technique can be used (e.g. UMAP), but the system is configured to read the projection coordinates from the dataframe by matching to the "layer_*_tsne_x" column format.
    - You will have to save this to disk either as a .csv or .dat file.
- Store the Attention Matrices:
    - This should be a dictionary with integer keys corresponding to the example id (this corresponds to the id in the DataFrame) and values should be another dictionary with the following keys: ('attn', 'token') and the values of this inner dictionary should be the attention matrices for the example and the list of tokens for the example respectively.
        - Example: {id: {'attn': (num_layers, num_heads, seq_len, seq_len)}, {'tokens': (seq_len)}}
    - You will want to save this to disk using `torch.save()` or `pickle.dump()` or some other object serialization method.
- Note the above steps can be done at the same time.

## How To Create A Task
1. Create the input collateral as described in the previous section (DataFrame and Attention Dict).
2. Define a new task class in `tasks.py` which inherits from either the `Task` abstract class or one of the other provided tasks classes, and add this task to the `get_task()` function in `tasks.py` (or alternatively use the one of the provided ones if your DataFrame is in the same format as the examples). There are 5 abstract methods that should be filled out in the new class, `_init_tsne_layout()`, `_init_table_cols_layout()`, `get_val_to_label_map()`, `get_tsne_rows()`, `get_summary_table`. These definitions will all depend on how the input DataFrame is structured.
    - When designing your own class, you will also see that there are 3 main types of functions associated with the class:
        - The `layout` functions will control the UI options (e.g. dropdown text, figure text, etc.)
        - The `map` functions will create dictionaries which map from internally used (by the application) keys to values provided in the collateral and by the user for convenience.
        - The `get` functions which are used by the application (`appConfiguration.py`) to access data held by the Task.

## Application Args:
- --collateral: either the path to the collateral pkl file or a space separated list of two paths if using the app in comparison mode.
- --df: either the path to the dataframe/csv file or a space separated list of two paths if using the app in comparison mode.
- --task: either "wsc" or "absa".
- --name: (Optional) name to be used for the model(s). 
- --num_layers: the number of layers in the model (defaults to 12).
- --port: specify the port to run the server on.

Normal mode (single model) example: 
- `python main.py --collateral "/path/to/collateral/model_collateral.pkl" --df "/path/to/dataframe/model_df.csv" --name "BERT" --task "wsc"  --num_layers 12 --port 5005 `

Comparison mode (two models) example: 
- `python main.py --collateral "/path/to/collateral/model1_collateral.pkl" "/path/to/collateral/model2_collateral.pkl" --df "/path/to/dataframe/model1_df.dat" "/path/to/dataframe/model2_df.dat" --name "my_model1" "my_model2" --task "absa"  --num_layers 12 --port 5005 `
