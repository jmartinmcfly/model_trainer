from __future__ import division
import json
import numpy as np
import math
import sys
import re

from sklearn import preprocessing
from sklearn.utils import shuffle
from keras.utils import np_utils

def transform_func(datapoint):
  '''
  Transforms the datapoint of a signal into a more effective representation.

  @param datapoint: a datapoint

  @return a transformed value
  '''
  
  return datapoint

def parse_json_accuracy(data, encoder, normalizer):
  '''
  Parses a series of data represented as a JSON object (mapping
  features to values) into a 2d numpy matrix that is formatted
  according to the expectation of the existing model and does the same for the labels.
  Maps features to columns using the encoder.

  @param data: The data to parse. Expected format {timestamp: {feature1: value, 
    feature2: value, ..., class: value}, ...}. Feature names can be anything.
  @param encoder: the object with the feature to column encoding
  @param normalizer: the object holding normalization values and functions

  @return: a 2d numpy array of data of shape (num_samples, num_addresses)

  @raises: ValueError
  '''
  try:
    parsed = json.loads(data)
  except ValueError:
    raise
  #list of datapoints. Each is a dict mapping features to values, 
  #and class to classNumber
  values = parsed
  #fill the array with data points
  #fill labels with string feature values (later one hot encoded)
  dataset = np.zeros((len(values), encoder.get_num_features()))
  labels = []
  for i, v in enumerate(values):
    for k in list(v.keys()):
      #fill data at sample i on the feature that the given feature maps to
      if k == "class":
          #fill label
          labels.append(v[k])
      elif k == "room":
        continue
      else:
        #fill data at sample i on the feature that the given feature maps to
        try:
          feature_index = encoder.encode_feature(k)
          dataset[i][encoder.encode_feature(k)] = transform_func(float(v[k]))
        except:
          # There is a new feature that was not seen in the training set.
          # Ignore it.
          continue
  #normalize
  dataset = normalizer.normalize(dataset)

  #one hot encode the labels and store the dictionary of encodings to labels
  #TODO: Fix this to use the method in encoder
  coded_labels = _encode_existing(labels, encoder.label_encoder)

  return dataset, coded_labels

def parse_json_classify(data, encoder):
  '''
  Parses a series of data represented as a JSON object (mapping
  features to values) into a 2d numpy matrix that is formatted
  according to the expectation of the existing model. Maps addresses to columns
  using the encoder.

  Parameters
  ----------
  data: The data to parse. Expected format {timestamp: {feature1: value, 
    feature2: value, ..., class: value}, ...}. Feature names can be anything.
  encoder: dictionary mapping features to columns

  Returns
  -------
  a 2d numpy array of data of shape (num_samples, num_addresses)

  Raises
  ------
  ValueError: If there is an error parsing the JSON
  '''
  try:
    parsed = json.loads(data)
  except ValueError:
    raise
  #list of datapoints. Each is a dict mapping feature to a value, 
  # room to roomName, and class to classNumber
  values = parsed
  #fill the array with data points
  dataset = np.zeros((len(values), encoder.get_num_features()))
  for i, v in enumerate(values):
    for k in list(v.keys()):
      #fill data at sample i on the feature that the given feature maps to
      if not (k == 'class'):
        try:
          feature_index = encoder.encode_feature(k)
          dataset[i][encoder.encode_feature(k)] = transform_func(float(v[k]))
        except:
          # There is a new feature that was not seen in the training set.
          # Ignore it.
          continue

  return dataset

def parse_json_new_model(data, encoder):
  '''
  Parses data (a JSON formatted string) into a feature set, a cleaned
  dataset, and a label set. Use this method to parse data for training a NEW
  model.

  Parameters
  ----------
  data: the json formatted string representing the dataset. Consists of
    a Json Object where the timestamp of each datapoint serves as the key. Can
    also be viewed as a list where the timestamps are the index. 

  Returns
  -------
  data: a numpy array with shape (num_samples, num_features). 
  encoder: an encoder object to populate with encodings
  
  Raises
  ------
  ValueError
  '''
  assert isinstance(data, str), "data is not a string"

  try:
    parsed = json.loads(data)
  except ValueError:
    raise
  #list of datapoints. Each is a dict mapping features to a values, 
  # and a class to classNumber
  values = parsed
  feature_to_index = feature_to_col(values)

  #fill the array with data points
  #fill labels with feature names (later one hot encoded)
  dataset = np.zeros((len(values), len(feature_to_index)))
  labels = np.zeros(len(values))
  #TODO: Get dummy dataset
  for i, v in enumerate(values):
    for k in list(v.keys()):
      if k == "class":
        #fill label
        labels[i] = v[k]
      else:
        #fill data at sample i on the feature that the given feature maps to
        dataset[i][feature_to_index[k]] = float(v[k])

  encoder = encoder.add_feature_encoding(feature_to_index)

  return dataset, labels, encoder

def feature_to_col(values):
  '''
  Calculates a mapping from a feature to a column in a 2d matrix.

  @param values
    a list of datapoints. Each is a dictionary mapping features to values.
  
  @return feature_to_index
    a dictionary mapping features to an index representing a column in the
    translated array
  '''
  feature_to_index = {}
  #extract the feature set. Dictionary of features 
  #to (soon to be) corresponding index in cleaned data.
  for v in values:
    for k in v.keys():
        if k != "class" and k != "room":
          feature_to_index.setdefault(k, len(feature_to_index))
  return feature_to_index

def _encode_existing(labels, encoding_dict):
  '''
  Encodes labels to one hot representation based on existing mapping. Expects
  that every label has an entry in the encoding dict and every 
  '''
  #assert that the encoding dict has enough labels. Doesn't check for a match.
  #assert(np.unique(np.asarray(labels)).shape[0] <= len(encoding_dict.keys()))
  if not labels:
    raise(ValueError("Labels is empty!"))
  encoded = np.zeros((len(labels), len(next(iter(encoding_dict.values())))))
  for i, l in enumerate(labels):
    # this is hacky.
    # TODO: Fix this by going into Model.training_pipeline and force the
    #       recording of the encoding to happen before we turn it to np array
    if float(l) in encoding_dict.keys():
      encoded[i][np.argmax(encoding_dict[float(l)])] = 1
  return encoded
    
