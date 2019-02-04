import numpy as np

from .normalizer import Normalizer

class EmptyNormalizer(Normalizer):
  '''
  This class contains information for normalizing data for a certain model.
  '''

  @abc.abstractmethod
  def fit(self, data):
    '''
    Fits the normalizer to the data
    '''
    return self

  @abc.abstractmethod
  def normalize(self, data):
    '''
    Normalizes the data. This will fail if the normalizer has not yet been fit.
    '''
    return data