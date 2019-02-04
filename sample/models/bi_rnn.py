from __future__ import division
import sys
if sys.version_info <= (3,0):
  import Queue as q
else:
  import queue as q

from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.callbacks import EarlyStopping
from .rnn import model_train_test_split
from sklearn.utils.extmath import softmax
import numpy as np

from .rnn import Rnn
from .dnn import Dnn
from sample.preprocessing.preprocessor import Preprocessor
from sample.preprocessing.categorical_encoder import Categorical_encoder
from sample.preprocessing.gaussian_normalizer import Gaussian_normalizer

class Bi_rnn(Rnn):
  '''
  This model is a recurrent Neural Network. It does not have to be stateful - I
  will attempt training it on batches of 3 and see if that helps.
  '''
  #the model type to be referred to in scripts
  MODEL_TYPE = 'bi_rnn'

  '''
  HYPERPARAMETERS
  '''
  #the function for transforming input values
  transformation = staticmethod(lambda x : (x + 100) % 100)
  #training params:
  # the number of epochs to train for (number of full passes over the data during training. Too
  #   few means underfitting, to many means overfitting)
  epochs = 100
  # the size of the validation set to hold out during training
  validation_size = 0.2
  # TODO: the architecture of the network. issue is it uses the data
  #architecture = np.asarray([layer_size, layer_size, num_labels])
  #default preprocessor
  preprocessor = Preprocessor(encoder=Categorical_encoder(), normalizer=Gaussian_normalizer(), 
    transformation=transformation)


  def __init__(self, model=None, preprocessor=None):
    if preprocessor == None:
      preprocessor = self.preprocessor
    super(Dnn, self).__init__(model=model, preprocessor=preprocessor)

  '''
  SECTION: Training
  '''
  
  def _train_single_model(self, X, Y, input_shape, architecture, verbose=False):
    '''
    Trains a model on data and Y (matched by index) and places the result 
    in model. 
    
    Parameters
    ----------
    X: a 2d numpy array containing unlabeled data points, with each row corresponding
      to a single data point, and each column corresponding to a feature. 
    Y: a 1d numpy array containing labels for the datapoints. Each label corresponds
      to the datapoint at the same row in data.
    input_dimension: the dimension of the input. Should be the number of MAC addresses measured.
    architecture: a numpy array containing the optimal architecture. Should allow for some
      automatic tuning.

    Returns
    -------
    the trained model
    '''
    batch = 32
    data_train, data_val, labels_train, labels_val = \
      model_train_test_split(X, Y, 0.2)

    #TODO: I need to write my own generators... the defaults won't do. My data
    #      is too weird.
    if input_shape[0] == None:
      length = 3
    else:
      length = input_shape[0]
    #TODO: Factor this into data preprocessing. Should be in _final_cleaning?
    #  Maybe _final_cleaning should include reshaping? Not sure how the generators
    #  factor in logically. Maybe they do fit in the training because I need certain
    #  info from np arrays that can't be as easily gotten from TimeSeriesGenerator.
    train_gen = TimeseriesGenerator(data_train, labels_train, length, batch_size=batch)
    val_gen = TimeseriesGenerator(data_val, labels_val, length, batch_size=batch)
    model = Sequential()
    #initialize input layer
    model.add(Bidirectional(LSTM(units=architecture[0], return_sequences=True, dropout=0.3, recurrent_dropout=0.3), input_shape=input_shape))
    #construct the rest of the architecture
    for i in range(1, len(architecture)):
      #we want a probability distribution over possible categories, so the
      #last layer should use softmax
      if i == len(architecture) - 1:
        model.add(Dense(units=architecture[i], activation='softmax'))
      elif i == len(architecture) - 2:
        model.add(Bidirectional(LSTM(units=architecture[i], dropout=0.3, recurrent_dropout=0.3)))
      else:
        model.add(Bidirectional(LSTM(units=architecture[i], return_sequences=True, dropout=0.3, recurrent_dropout=0.3)))
    #What are the best values for loss/optimizer/metrics? Need to research
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    
    es = EarlyStopping(monitor='val_loss',
                              min_delta=-0.1,
                              patience=5,
                              verbose=0, mode='auto')
    if verbose:
      self.history = model.fit_generator(train_gen, steps_per_epoch=len(data_train) / batch, epochs=self.epochs, 
        validation_data=val_gen, validation_steps=len(data_val) / batch, verbose=1, callbacks=[es])
    else:
      self.history = model.fit_generator(train_gen, steps_per_epoch=len(data_train) / batch, epochs=self.epochs, 
        validation_data=val_gen, validation_steps=len(data_val) / batch, verbose=0, callbacks=[es])
    self.model = model
    return model
