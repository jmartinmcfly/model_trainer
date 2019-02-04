import numpy as np

from .normalizer import Normalizer

class Gaussian_normalizer(Normalizer):
  '''
  A gaussian normalizer. Allows you to transform a dataset to one with an average
  of zero and a standard deviation of one. Useful for neural networks especially.
  '''
  average = 0
  variance = 1

  def __init__(self, average=0, variance=1):
    self.average = average
    self.variance = variance

  def fit(self, data):
    '''
    Fits the normalizer to the data. Should only be called once during the life of
    each normalizer object.
    '''
    self.average = np.mean(data, axis=0)
    self.variance = np.std(data, axis=0)
    return self

  def normalize(self, data):
    '''
    Normalizes the data
    '''
    variance = self.variance
    # Protect against divide by 0
    variance[variance == 0] = 1
    normalized = (data - self.average) / variance
    return normalized