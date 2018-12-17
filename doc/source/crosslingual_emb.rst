Unsupervised Crosslingual Embeddings
####################################

Overview
========
This model uses a GAN to learn mapping between two language embeddings without supervision as demonstrated in Word Translation Without Parallel Data [1]_.

.. image:: assets/w2w.png


Files
=====
- **nlp_architect/data/fasttext_emb.py**: Defines Fasttext object for loading Fasttext embeddings
- **nlp_architect/models/crossling_emb.py**: Defines GAN for learning crosslingual embeddings
- **examples/crosslingembs/train.py**: Trains the model and writes final crosslingual embeddings to weight_dir directory.
- **examples/crosslingembs/evaluate.py**: Defines graph for evaluating the quality of crosslingual embeddings

Usage
=====
Main arguments which need to be passed to train.py are

- **emb_dir**: Directory where Fasttext embeddings are present or need to be downloaded
- **eval_dir**: Directory where evaluation dictionary is downloaded
- **weight_dir**: Directory where final crosslingual dictionaries are defined

Use the following command to run training and generate crosslingual embeddings file:

.. code:: python

  python train.py --data_dir <embedding dir> --eval_dir <evaluation data> \
    --weight_dir <save_data> --epochs 1

Example Usage
---------------

Make directories for storing downloaded embeddings and multi language evaluation dictionaries

.. code:: bash

  mkdir data
  mkdir ./data/crosslingual/dictionaries

Run training sequence pointing to embedding directory and multi language evaluation dictionaries. After training it will store the mapping weight and new cross lingual embeddings in weight_dir

.. code:: python

  python train.py --data_dir ./data --eval_dir ./data/crosslingual/dictionaries --weight_dir ./

Results
=======

When trained on English and French embeddings the results for word to word translation accuracy are as follows

.. csv-table::
  :header: "Eval Method ",K=1, K=10
  :widths: 25, 20, 20
  :escape: ~

  NN,53.0,74.13
  CSLS,81.0, "93.0 "


References
==========
.. [1] Alexis Conneau, Guillaume Lample, Marc’Aurelio Ranzato, Ludovic Denoyer, Herve Jegou Word Translation Without Parallel Data https://arxiv.org/pdf/1710.04087.pdf
.. [2] P.Bojanowski, E. Grave, A. Joulin, T. Mikolov, Enriching Word Vectors with Subword Information https://arxiv.org/abs/1607.04606
