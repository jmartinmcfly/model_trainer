
from .gaussian_normalizer import Gaussian_normalizer
from .categorical_encoder import Categorical_encoder

class Preprocessor():
  '''
  This class is a wrapper for an encoder (to encode inputs to a form the model can
  process as well as encode/decode outputs from model speak to broader system speak)
  and a normalizer (to normalize the data).
  '''
  encoder = None
  normalizer = None
  transformation = None

  def __init__(self, encoder=Categorical_encoder(), normalizer=Gaussian_normalizer(), 
    transformation=lambda x : x):
    self.encoder = encoder
    self.normalizer = normalizer
    self.transformation = transformation

  def get_encoder(self):
    return self.encoder
  
  def get_normalizer(self):
    return self.normalizer

  def get_transformation(self):
    return self.transformation