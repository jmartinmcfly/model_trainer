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
from sklearn.model_selection import train_test_split
from sklearn.utils.extmath import softmax
import numpy as np

from .dnn import Dnn
from sample.preprocessing.preprocessor import Preprocessor
from sample.preprocessing.categorical_encoder import Categorical_encoder
from sample.preprocessing.gaussian_normalizer import Gaussian_normalizer

class Rnn(Dnn):
  '''
  This model is a recurrent Neural Network. It does not have to be stateful - I
  will attempt training it on batches of 3 and see if that helps.
  '''
  #the model type to be referred to in scripts
  MODEL_TYPE = 'rnn'

  '''
  HYPERPARAMETERS
  '''
  #the function for transforming input values
  transformation = staticmethod(lambda x : (x + 100) % 100)
  #training params:
  # the number of epochs to train for (number of full passes over the data during training. Too
  #   few means underfitting, to many means overfitting)
  epochs = 300
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

  #TODO: Factor out input_dimension or at least set a default = data.shape[0]
  def select_model(self, X, Y, verbose=False, input_shape=-1):
    '''
    Automatically finds the optimal hyperparameters and architecture for the
    NN.

    Parameters
    ----------
    X: data points
    Y: labels matched by index to the datapoints
    input_shape: the shape of each input vector (timesteps, input_dim).
      timesteps can be None if they vary.
    num_labels: the number of unique labels
    verbose: Whether or not to print verbose X from training. Default not verbose.
    '''
    if input_shape == -1:
      #hack because I cannot access X.shape in the declaration of arguments for setting the default
      input_shape = (None, X.shape[1])
    #can do this because of the one hot encoding
    num_labels = Y.shape[1]
    layer_size = np.int(input_shape[1] * 1.5)
    architecture = np.asarray([layer_size, layer_size, num_labels])
    #make it more clear which is input, hidden, and output
    self.model = self._train_single_model(X, Y, input_shape, architecture, verbose=verbose)
  
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
    input_dimension: the dimension of the input.
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
    model.add(LSTM(units=architecture[0], return_sequences=True, dropout=0.3, recurrent_dropout=0.3, input_shape=input_shape))
    #construct the rest of the architecture
    for i in range(1, len(architecture)):
      #we want a probability distribution over possible categories, so the
      #last layer should use softmax
      if i == len(architecture) - 1:
        model.add(Dense(units=architecture[i], activation='softmax'))
      elif i == len(architecture) - 2:
        model.add(LSTM(units=architecture[i], dropout=0.3, recurrent_dropout=0.3))
      else:
        model.add(LSTM(units=architecture[i], return_sequences=True, dropout=0.3, recurrent_dropout=0.3))
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

  def _accuracy_n(self, X, Y, group_size=1, seq_size=3):
    '''
    Calculates the accuracy of the model where a correct classification is defined
    as predicting the correct label within the n most likely categories. The
    input is segmented into disjoin sequences of length window, and each
    prediction is per sequence.

    Parameters
    ----------
    X: a 2d numpy array of inputs
      the inputs we are attempting to classify
    Y: a 2d numpy array of one hot encodings
      the corresponding labels
    group_size: The number of predictions that go into any larger prediction
    pred_size: the margin for error. A prediction will be considered correct
      if the correct answer is within the first group_size guesses. group_size = 1
      is equivalent to keras' normal accuracy metric.

    Returns
    -------
    The proportion of examples correctly classified within the first n guesses.
    Does not take into account relative confidence.
    '''
    
    slack = seq_size - (X.shape[0] % seq_size)
    if slack == 0:
      pass
    else:
      # stack repeat inputs onto the end so that the array divides evenly
      # by seq_size
      padding_inputs = X[-slack:]
      padding_labels = Y[-slack:]
      X = np.vstack((padding_inputs, X))
      Y = np.vstack((padding_labels, Y))
      if not X.shape[0] % seq_size == 0:
        assert(X.shape[0] % seq_size == 0), 'Error on padding data'
      samples = X.shape[0] // seq_size
    assert(X.shape[0] == Y.shape[0])
    
    #TODO: Finish this
    # modify X to the format first, second, third, second, third, fourth, third, fourth, fifth, ...
    # and modify Y to match indexwise
    # Idea is to see the classification on each subsequent datapoint given all previous data
    arr_size = (X.shape[0] - (seq_size - 1)) * seq_size
    xs = np.zeros((arr_size, seq_size, X.shape[1]))
    ys = np.zeros((arr_size, seq_size, Y.shape[1]))
    for i in range(0, X.shape[0] - (seq_size - 1)):
      for j in range(seq_size):
        xs[i][j] = X[i + j]
        ys[i][j] = Y[i + j]

    assert(xs.shape[0] == ys.shape[0])

    # reshape into sequences of size seq_size
    #xs = xs.reshape(
    #    (xs.shape[0] / seq_size, seq_size, xs.shape[1]))
    #ys = ys.reshape(ys.shape[0] / seq_size, seq_size, Y.shape[1])
    
    predictions = self.model.predict(xs)

    correct = 0
    for i in range(0, len(predictions)):
      # sorts the array, replaces the values with their original indices, and
      # then slices out everything but the first group_size elements of the array
      pred = predictions[i]

      arr = np.argsort(-pred)[:group_size]
      # if the one hot encoded correct label (represented by its significant index)
      # matches any of the values in arr, then we correctly classified the group
      if np.any(np.argmax(ys[i][seq_size - 1]) == arr):
        correct += 1
    
    return correct / len(predictions)

  def accuracy_accum(self, X, Y, window=1, group_size=1):
    '''
    Calculates the accuracy of the model where a correct classification is defined
    as predicting the correct label within the n most likely categories. The
    input is segmented into overlapping sequences of length window, and each
    prediction is per sequence. This mirrors real world application where we
    generate a new prediction on every new datapoint (using the previous window - 1 inputs
    to fill out the sequence).

    Parameters
    ----------
    X: a 2d numpy array of inputs
      the inputs we are attempting to classify
    Y: a 2d numpy array of one hot encodings
      the corresponding labels
    group_size: The number of predictions that go into any larger prediction
    pred_size: the margin for error. A prediction will be considered correct
      if the correct answer is within the first group_size guesses. group_size = 1
      is equivalent to keras' normal accuracy metric.

    Returns
    -------
    The proportion of examples correctly classified within the first n guesses.
    Does not take into account relative confidence.
    '''
    samples = np.zeros((len(X) - window, window, X.shape[2]))
    for i in range(0, len(X) - window):
      for j in range(i, i+window):
        samples[i][j - i] = X[j]
    
    X = samples
    
    predictions = self.model.predict(X)

    correct = 0
    for i in range(0, len(predictions)):
      # sorts the array, replaces the values with their original indices, and
      # then slices out everything but the first group_size elements of the array
      pred = predictions[i]

      arr = np.argsort(-pred)[:group_size]
      # if the one hot encoded correct label (represented by its significant index)
      # matches any of the values in arr, then we correctly classified the group
      label_index = (window - i) + i * window
      if np.any(np.argmax(Y[label_index]) == arr):
        correct += 1
    
    return correct / len(predictions)

  @staticmethod
  def load_saved_model(path):
    '''
    Loads the model of the given name at the path.

    Parameters
    ----------
    path: a path to the folder holding the model. Should be of the form
      'path_to_all_models/model_name'
    
    Returns
    -------
    model: The loaded raw model
    preprocessor: The preprocessor used in to train the model
    
    Raises
    ------
    ValueError: if the model save file is invalid
    ImportError: if HP5Y is not installed
    '''
    preprocessor = super(Dnn, Dnn).load_saved_model(path)

    try:
      model = load_model(path + 'model.h5')
    except ValueError:
      raise ValueError("The model savefile is invalid.")
    except ImportError:
      raise ImportError("HP5Y is not available. It is needed to decode the model save file. " +
      "See installation instructions at http://docs.h5py.org/en/latest/build.html.")
    # NOTE: This works for the bi_rnn as well which is why it doesn't have to
    #       be overloaded there
    return Rnn(model, preprocessor)

  def classify(self, sequence, window_size=10, group_size=1, verbose=False):
    '''
    Generates a prediction on a set of datapoints. Treats them as a single
    sequence.

    Parameters
    ----------
    sequence: a sequence of inputs to classify as a single group

    Returns
    -------
    prediction: the most likely class
    confidence: the likelihood of the class
    '''
    #TODO: Implement window_size
    #TODO: Implement verbose
    #TODO: Implement group_size
    #TODO: Fix this hacky shit
    prediction = []
    conf = []
    sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
    raw_prediction = self.model.predict(sequence)[0]
    arr = np.argsort(-raw_prediction)[:group_size]
    for ele in arr:
      prediction.append(self.preprocessor.get_encoder().decode_label(ele))
      conf.append(raw_prediction[ele])
    
    return prediction, conf

def model_train_test_split(X, Y, holdout):
  #TODO: Add randomization if possible
  '''
  Does a train test split for sequence data, maintaining sequence integrity. Note
  that it stacks classes in most recently seen, first in array.

  Parameters
  ----------
  X: the datapoints
  Y: labels, matched by index to data
  holdout: the proportion of the full set to hold out for testing/validation

  Returns
  -------
  x_train: 1 - holdout proportion of data
  x_test: holdout proportion of data
  y_train: labels for x_train
  y_test: labels for x_test
  '''
  for i in range(len(Y[0])):
    x = X[Y[:, i] == 1]
    y = Y[Y[:, i] == 1]
    size = int(len(x) * holdout)
    if i == 0:
      x_test = x[0:size]
      x_train = x[size:]
      y_test = y[0:size]
      y_train = y[size:]
    else:
      x_test = np.vstack((x[0:size], x_test))
      x_train = np.vstack((x[size:], x_train))
      y_test = np.vstack((y[0:size], y_test))
      y_train = np.vstack((y[size:], y_train))
  return x_train, x_test, y_train, y_test

def variable_length_generator(X, Y, min_len, max_len, batch_size):
  '''
  This is a generator that yields variable length sequences from [min_len, max_len],
  sampling uniformly. Assumes that the data is structured chronoligcally and
  stratified by class.

  Parameters
  ----------
  X: the datapoints in chronological order, grouped by label
  Y: the label of each datapoint
  min_len: minimum length of a yielded sequence
  max_len: maximum length of a yielded sequence
  batch_size: the size of each yielded batch

  Returns
  -------
  A numpy array of shape (batch_size, [min_len, max_len], features) containing
  '''
  #TODO: write variable_length_generator.
  pass
  
