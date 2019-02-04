'''
Container for model_list
'''
from .dnn import Dnn
from .rnn import Rnn
from .bi_rnn import Bi_rnn
from .random_forest import Random_forest

models = {
  'dnn' : Dnn,
  'rnn' : Rnn,
  'bi_rnn' : Bi_rnn
  #'cnn' : cnn
  #'random_forest' : Random_forest #,
  #'svm' : svm,
  #'naive_bayes' : n_bayes
}

def get_models():
  return models