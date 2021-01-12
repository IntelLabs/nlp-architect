# InterpreT: An Interactive Visualization Tool for Interpreting Transformers
## Part of NLP Architect by Intel® AI Lab

## Overview

As Transformers are increasingly gaining widespread use for NLU/NLP tasks, there has been growing interest in understanding the inner workings of these models and why they are so effective at such tasks. To further this goal of explainability and comprehension, we present InterpreT. Our system is a general tool for facilitating understanding of the behaviors of Transformer models, and we demonstrate our system's functionality through analyzing model behavior for two disparate tasks: the Winograd Schema Challenge (WSC) and Aspect Based Sentiment Analysis (ABSA). In addition to providing various mechanisms for investigating general model behaviors, InterpreT enables novel, granular analysis through the layer-level probing and visualization of internal representations of Transformer models, allowing users to gain new insights into how and what their models are learning. 

## InterpreT Demo Website
A demo of InterPret on the WSC task can be accessed at the following link: http://interpret.intel-research.net/

By analyzing corereference resolution through InterpreT, we demonstrate that a fine-tuned BERT model pushes closer together the embeddings of terms it predicts to be coreferent. The metric "finetuned_coreference_intensity" in the Head Summary plot shows that the 7th head of layer 10 often attends between coreferent mention spans. This attention head can also be visualized in InterpreT for various examples. 

## Screencast Video Demo

[![Video Demo](https://raw.githubusercontent.com/IntelLabs/nlp-architect/master/solutions/absa_solution/assets/video.png)](https://drive.google.com/file/d/1MbESn2RI58PYsfhX4zX9jivCzQzuhtPR/view)
*Figure 1*
