# Training a Model #

## Introduction ##

In this document, I will briefly explain my rationale through which I structured this repoistory. This document will not delve into the specifics of any given model. Rather, it will attempt to lay out a broader design for dealing with data, a so called *data pipeline*. A data pipeline consists of several parts:

1. Collecting and storing the data
2. Feature engineering (modifying and adding features)
3. Feature selection (dimensionality reduction)
4. Model training
5. Evaluation
6. Deployment and in production predictions

This repository deals with steps two through five. Note that the pipeline can loop between steps four and five before deciding on an optimal model / architecture (a process called model selection). I encapsulated steps 
two and three in the preprocess_train method, step 4 and 5 in the select_model method, and step 6 in the classify method (do note that much of the production logic, such as receiving data from sensors and choosing the correct stored model lay outside of the scope of this repository).

---

## Processing the Data ##

### Parsing ###

### Feature Engineering ###

### Training, Validation, and Testing Sets ###

---

## Tuning the Model ##

### Meta Parmeters: the Architecture ###

### Hyperparameters ###

Batch size. Learning rate. Number of Epochs.

### Regularization ###

---

## Visualization ##
