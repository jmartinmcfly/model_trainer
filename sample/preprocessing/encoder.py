import abc
import numpy as np

class Encoder(object):
  '''
  This class contains information for encoding/decoding data for a certain model.
  This class allows you to encode datapoints and encode/decode labels.
  '''
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def add_feature_encoding(self, feature_encoding):
    '''
    Allows you to set a feature encoding that has already been fit.
    '''
    pass
  @abc.abstractmethod 
  def _add_label_encoding(self, encoding_dict, decoding_dict):
    '''
    Adds the label encoding to the object's state
    '''
    return self

  @abc.abstractmethod
  def fit_labels(self, Y):
    '''
    Takes unencoded labels and returns their encoded version.
    '''
    pass

  @abc.abstractmethod
  def encode_feature(data):
    '''
    Maps feature names to column numbers
    '''
    pass

  @abc.abstractmethod
  def encode_label(labels):
    '''
    Maps original label to that used in the machine learning model.
    '''
    pass

  @abc.abstractmethod
  def decode_label(labels):
    '''
    Maps the encoded label (used in the ML model) to the original label.
    '''
    pass

  @abc.abstractmethod
  def get_num_features():
    '''
    Returns the number of features in the data that the encoder encodes.
    '''
    pass

  @abc.abstractmethod
  def get_num_labels():
    '''
    Returns the number of labels in the data that the encoder encodes.
    '''
    pass

  def encode_existing(self, labels, encoder):
    '''
    Encodes a list of labels with the mapping held in encoder.
    '''
    coded_labels = np.zeros(len(labels))
    for i, l in enumerate(labels):
      coded_labels[i] = encoder.encode_label(i)