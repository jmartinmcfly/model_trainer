from .encoder import Encoder

import numpy as np
from keras.utils import np_utils

class Categorical_encoder(Encoder):
  '''
  An encoder for categorical data. Expects a one hot encoding.
  '''
  label_encoder = {}
  label_decoder = {}
  data_encoder = {}
  immutability_counter = 0

  #TODO: Should I define a build method?

  def add_feature_encoding(self, feature_encoding):
    '''
    Allows you to set a feature encoding that has already been fit.
    '''
    self.data_encoder = feature_encoding
    return self

  def fit_labels(self, Y):
    '''
    Takes unencoded labels and returns their encoded version.

    Parameters
    ----------
    Y: The raw labels, a numpy array of shape (samples,)

    Returns
    ------
    An numpy array of one hot encodings of shape (samples, num_labels)
    '''
    if len(Y) == 0:
      return [], self
    #encode to one_hot and create the encoding dictionary
    #ids encodes each label down to its corresponding index in uniques. We need
    #this integer encoding to pass to to_categorical
    uniques, ids = np.unique(Y, return_inverse=True)
    coded_labels = np_utils.to_categorical(ids, len(uniques))
    
    encoding_dict = {}
    decoding_dict = {}
    #generate the encoding dictionary (integer encoding -> one hot)

    #NOTE: This goes over the whole array (so lots of duplicates). Would it be
    # quicker to to np.unique first?
    #TODO: Make this more efficient
    for i,j in zip(Y, coded_labels):
      encoding_dict[i] = j
      #Note that this maps the index that defines the one hot encoding to
      #the original label (the key is an index, not a vector)
      decoding_dict[np.argmax(j)] = i
      if len(encoding_dict)==len(uniques):
          break

    encoder = self._add_label_encoding(encoding_dict, decoding_dict)

    return coded_labels, encoder

  def _add_label_encoding(self, encoding_dict, decoding_dict):
    '''
    Adds the label encoding to the object's state
    '''
    self.label_encoder = encoding_dict
    self.label_decoder = decoding_dict
    return self

  def encode_feature(self, col_name):
    '''
    Maps feature names to column numbers
    Raises an error if the feature is not encoded (aka was not in the training set)
    '''
    if col_name in self.data_encoder.keys():
      return self.data_encoder[col_name]
    else:
      raise ValueError("The feature is not in the list of encodings")
  
  def encode_label(self, raw_label):
    '''
    Maps original label to that used in the machine learning model.
    '''
    return self.label_encoder[raw_label]

  def decode_label(self, encoded_label):
    '''
    Maps the encoded label (used in the ML model) to the original label.
    '''
    return self.label_decoder[np.argmax(encoded_label)]

  def get_num_features(self):
    '''
    Returns the number of features in the data that the encoder encodes.
    '''
    return len(self.data_encoder.keys())
  
  def get_num_labels(self):
    '''
    Returns the number of labels in the data that the encoder encodes.
    '''
    return len(self.label_encoder.keys())
  
  #TODO: Should I build the encoding here?