# Supervised Sentiment

This is a set of models which are examples of supervised implementations for sentiment analysis.
The larger idea behind these models is to allow ensembling with other supervised or unsupervised models.


# Models
Two models are shown as classification examples. Additional models can be added as desired.

<b>Bi-directional LSTM</b>
A simple bidirectional lstm with one fully connected layer. The number of vocab features, dense output size, and document input length, should be determined in the data preprocessing steps. The user can then change the size of the lstm hidden layer, and the recurrent dropout rate.

<b>Temporal CNN</b>
As defined in "Text Understanding from Scratch" by Zhang, LeCun 2015 https://arxiv.org/pdf/1502.01710v4.pdf this model is a series of 1D CNNs, with a maxpooling and fully connected layers. The frame sizes may either be large or small.


# Datasets
The dataset in this example is the Amazon Reviews dataset, though other datasets can be easily substituted.
The Amazon review dataset(s) should be downloaded from http://jmcauley.ucsd.edu/data/amazon/. These are `*.json.gzip` files which should be unzipped. The terms and conditions of the data set license apply. Intel does not grant any rights to the data files.
For best results, a medium sized dataset should be chosen though the algorithms will work on larger and smaller datasets as well. For experimentation I chose the Movie and TV reviews.
Only the "overall", "reviewText", and "summary" columns of the review dataset will be retained. The "overall" is the overall rating in terms of stars - this is transformed into a rating where currently 4-5 stars is a positive review, 3 is neutral, and 1-2 stars is a negative review.
The "summary" or title of the review is concatenated with the review text and subsequently cleaned.

The Amazon Review Dataset was published in the following papers:

Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering
R. He, J. McAuley
WWW, 2016
http://cseweb.ucsd.edu/~jmcauley/pdfs/www16a.pdf

Image-based recommendations on styles and substitutes
J. McAuley, C. Targett, J. Shi, A. van den Hengel
SIGIR, 2015
http://cseweb.ucsd.edu/~jmcauley/pdfs/sigir15.pdf


# Train/Test
Currently, the pipeline shows a full train/test/ensemble cycle. The main pipeline can be run with the following command:
```
 python example_ensemble.py --file_path ./reviews_Movies_and_TV.json/
```
At the conclusion of training a final confusion matrix will be displayed.

# Hyperparameter optimization
An example of hyperparameter optimization is given using the python package hyperopt which uses a Tree of Parzen estimator to optimize the simple bi-lstm algorithm. To run this example the following command can be utilized:
```
 python optimize_example.py --file_path ./reviews_Movies_and_TV.json/ --new_trials 50 --output_file ./data/optimize_output.pkl
```
The file will output a result of each of the trial attempts to the specified pickle file.
