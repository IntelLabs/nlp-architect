# UNSUPERVISED CROSSLINGUAL EMBEDDINGS
This model learns crosslingual embedding in an unsupervised manner using GANs as demonstrated in
Word Translation Without Parallel Data by Alexis Conneau et al.,


Use the following command to run training and generate crosslingual embeddings file

```python train.py --data_dir <embedding dir> --eval_dir <evaluation data> --weight_dir <save_data> --epochs 1```

Example Usage
```mkdir data```
```mkdir ./data/crosslingual/dictionaries```
```python train.py --data_dir ./data --eval_dir ./data/crosslingual/dictionaries --weight_dir ./ ```

 Citations
  ---------
  1. Alexis Conneau, Guillaume Lample, Marc’Aurelio Ranzato, Ludovic Denoyer, Herve Jegou Word Translation Without Parallel Data https://arxiv.org/pdf/1710.04087.pdf
  2. P. Bojanowski, E. Grave, A. Joulin, T. Mikolov, Enriching Word Vectors with Subword Information https://arxiv.org/abs/1607.04606
  