import abc

class Normalizer(object):
  '''
  This class contains information for normalizing data for a certain model.
  '''
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def fit(self, data):
    '''
    Fits the normalizer to the data
    '''
    pass

  @abc.abstractmethod
  def normalize(self, data):
    '''
    Normalizes the data. This will fail if the normalizer has not yet been fit.
    '''
    pass