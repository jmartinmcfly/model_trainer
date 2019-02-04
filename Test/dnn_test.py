import pytest
import os
import json

import numpy as np

from sample.models.dnn import Dnn
from sample.preprocessing.preprocessor import Preprocessor
from sample.preprocessing.categorical_encoder import Categorical_encoder
from sample.preprocessing.gaussian_normalizer import Gaussian_normalizer



def test_dnn_accuracy_n():
  dummy = Dummy(3)
  model = Dnn(model=dummy)
  inputs = np.asarray([
    [1, 2],
    [3, 4],
    [5, 6],
    [1, 2]
  ])
  labels = np.asarray([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0]
  ])
  results = model._accuracy_n(inputs, labels)
  assert results == 1.0
  labels = np.asarray([
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0]
  ])
  assert results 

@pytest.mark.parametrize("data", [
  ("data/iris.json")])

def test_dnn_preprocessing(data):
  #takes in json formatted inputs and a dict mapping features to columns
  model = Dnn()
  data = open(data).read()
  preprocessor = Preprocessor(Categorical_encoder(), 
    Gaussian_normalizer(), Dnn.transformation)
  X, Y, preprocessor = model.preprocess_train(data, preprocessor)
  encoder = preprocessor.get_encoder()
  normalizer = preprocessor.get_normalizer()
  #check that the shape mac_to_index, dataset, coded_labels, and encoding_dict
  #all match
  feature_dict = encoder.data_encoder
  encoding_dict = encoder.label_encoder
  decoding_dict = encoder.label_decoder
  norm_avg = normalizer.average
  norm_var = normalizer.variance
  assert X.shape[1] == len(feature_dict.keys())
  assert X.shape[0] == Y.shape[0]
  assert norm_avg.shape[0] == X.shape[1]
  assert norm_var.shape[0] == X.shape[1]
  print(Y)
  assert np.unique(Y, axis=0).shape[0] == len(encoding_dict.keys())
  
  X_train, X_test, Y_train, Y_test = model.model_train_test_split(X, Y, test_size=0.5)

  #TODO: implement this test. Need to internalize parsing (issue is x_train is already normalized)
  X_test = model.preprocess_classify(X_train, normalizer, Dnn.transformation)
  #assert np.all(X_train == X_test)

def test_dnn_evaluate():
  pass

def test_dnn_save_model():
  #can I test this?
  pass

class Dummy():
  size = None

  def __init__(self, size):
    self.size = size
  
  def predict(self, inputs):
    predictions = np.zeros((inputs.shape[0], self.size))
    for count, i in enumerate(inputs):
      predictions[count][i[0] % self.size] = 1
    
    return(predictions)