from __future__ import division

import sys
import Queue as q
import pickle
import os
import math

import dill
import keras
from keras.optimizers import Adam
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.utils.extmath import softmax
import numpy as np

from sample.parsing.parser import parse_json_new_model as parse_train
from sample.parsing.parser import parse_json_classify as parse_classify
from sample.preprocessing.preprocessor import Preprocessor
from sample.preprocessing.gaussian_normalizer import Gaussian_normalizer
from sample.preprocessing.categorical_encoder import Categorical_encoder
from .model import Model, unstaticmethod

'''
The file is organized preprocessing -> training -> evaluation -> classification. 
All non stateful helper methods (of the form _helper_name) are placed at the
end of the file, outside of the scope of Dnn.
'''

class Dnn(Model):
  '''
  This class encapsulaters a model (a Keras Neural Net) to classify using a dense neural net.
  '''
  #the model type to be referred to in scripts
  MODEL_TYPE = 'dnn'
  #the history (training loss and validation loss by epoch) from training of the model
  history = None
  #the accumulated prediction
  accumulated_pred = None
  #the last predictions stored as a queue - you decrement accumulated by the
  #last seen prediction
  last_predictions = q.Queue()
  #the identity function. Can be customized
  transformation = staticmethod(lambda x : x)
  preprocessor = Preprocessor(encoder=Categorical_encoder(), normalizer=Gaussian_normalizer(),
    transformation=transformation)
  #training params
  epochs = 110

  def __init__(self, model=None, preprocessor=None):
    if preprocessor == None:
      preprocessor = self.preprocessor
    super(Dnn, self).__init__(model=model, preprocessor=preprocessor)
    self.accumulated_pred = np.zeros(preprocessor.get_encoder().get_num_labels())

  '''
  SECTION: Training
  '''

  @staticmethod
  def _final_cleaning(X, Y):
    '''
    Final cleaning. Removes shitty data (in this case, dupes).
    NOTE: Should this be part of feature engineering?

    Parameters
    ----------
    X: data
    Y: labels, matched by index

    Returns
    -------
    X, Y with duplicates removed
    '''
    #remove duplicates
    #X, indices = np.unique(X, axis=0, return_index=True)
    #Y = Y[indices]

    return X, Y

  def select_model(self, X, Y, verbose=False, validation_size=0.01):
    '''
    Automatically finds the optimal hyperparameters and architecture for the
    NN.

    Parameters
    ----------
    X: data points
    Y: labels matched by index to the datapoints
    verbose: Whether or not to print verbose data from training. Default not verbose.
    validation_size: The size to use for the validation set during training.

    Returns
    -------
    the trained model
    '''
    input_dimension = X.shape[1]
    #can do this because of the one hot encoding
    num_labels = Y.shape[1]
    layer_size = np.int(input_dimension * 1.2)
    layer_size = 1200
    intro = np.int(input_dimension * 0.75)
    architecture = np.asarray([layer_size, layer_size, num_labels])
    self.model = self._train_single_model(X, Y, input_dimension, architecture, validation_size,
     verbose=verbose)

  def _train_single_model(self, X, Y, input_dimension, architecture, validation_size,
    verbose=False):
    '''
    Trains a model on data and labels (matched by index) and places the result 
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
    validation_size: The proportion of X to hold out for validation
    verbose: Whether or not to print verbose training metadata.

    @return the trained model
    '''

    data_train, data_val, labels_train, labels_val = \
      train_test_split(X, Y, test_size=validation_size)

    model = Sequential()
    #initialize input layer
    model.add(Dense(units=architecture[0], activation='relu', input_dim=input_dimension))
    #construct the rest of the architecture
    for i in range(1, len(architecture)):
      #we want a probability distribution over possible categories, so the
      #last layer should use softmax
      if i == len(architecture) - 1:
        model.add(Dense(units=architecture[i], activation='softmax'))
      else:
        model.add(Dense(units=architecture[i], activation='relu'))
        if i != 0:
          #add dropout to help prevent overfitting
          model.add(Dropout(0.2))
    #What are the best values for loss/optimizer/metrics? Need to research
    #opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=False)
    opt = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    model.compile(loss='categorical_crossentropy',
                  #optimizer='rmsprop',
                  optimizer=opt,
                  metrics=['acc'])
    
    es = EarlyStopping(monitor='val_loss',
                              min_delta=-0.1,
                              patience=5,
                              verbose=0, mode='auto')


    scheduler = LearningRateScheduler(exp_decay)
    if verbose:
      #self.history = model.fit(data_train, labels_train, epochs=self.epochs, batch_size=32, 
       # validation_data=(data_val, labels_val), verbose=1, callbacks=[es])
        self.history = model.fit(data_train, labels_train, epochs=self.epochs, batch_size=32, 
        validation_data=(data_val, labels_val), verbose=1, callbacks=[es])
    else:
      self.history = model.fit(data_train, labels_train, epochs=self.epochs, batch_size=32, 
        validation_data=(data_val, labels_val), verbose=0, callbacks=[es])
    self.model = model
    return model

  def save_model(self, filepath):
    '''
    Saves the model at filepath as an HDF5. Expects filepath to end in .h5.

    Parameters
    ----------
    filepath: A filepath
    '''
    self.model.save(filepath + 'model.h5')

  '''
  SECTION: Evaluation
  '''

  def _accuracy_n(self, X, Y, group_size=1, seq_size=1):
    #TODO: Test this - it's being kinda funky
    '''
    Calculates the accuracy of the model where a correct classification is defined
    as predicting the correct label within the n most likely categories.

    Parameters
    ----------
    X: a 2d numpy array of inputs
      the inputs we are attempting to classify
    Y: a 2d numpy array of one hot encodings
      the corresponding labels
    group_size: the margin for error. A prediction will be considered correct
      if the correct answer is within the first group_size guesses. group_size = 1
      is equivalent to keras' normal accuracy metric.
    seq_size: The number of predictions that go into any larger prediction

    Returns
    -------
    The proportion of examples correctly classified for each sequence of size seq_size
    within the first n guesses. Does not take into account relative confidence.
    '''

    predictions = self.model.predict(X)
    correct = 0
    for i in range(0, len(predictions), 1):
      #sorts the array, replaces the values with their original indices, and
      #then slices out everything but the first group_size elements of the array
      preds = np.zeros(predictions[i].shape[0])
      for j in range(i, i + seq_size):
        if j < len(predictions):
          preds += predictions[j]

      arr = np.argsort(-preds)[:group_size]
      #if the one hot encoded correct label (represented by its significant index)
      #matches any of the values in arr, then we correctly classified the group
      if np.any(np.argmax(Y[i]) == arr):
        correct += 1
    
    return correct / len(X)

  '''
  SECTION: Classification
  '''

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
    
    #TODO: Consider factoring out loading/saving logic into encoder/normalizer class
    #      Only issue is with loading logic... need to know which subclass of encoder
    #      and normalizer we are dealing with
    
    return Dnn(model, preprocessor)

  def classify(self, X, window_size=10, group_size=1, verbose=False):
    #TODO: Factor predictions into a prediction class, with two entries - prediction, and confidence
    '''
    Predicts the group_size most likely classes given a set of data. Assumes
    that the classes are represented as integers.

    Parameters
    ----------
    X: A numpy matrix of vectors. Expects shape (#samples,#features)
    window_size: The number of datapoints to take into account for the final prediction.
    group_size: The number of classes to return for each datapoint. If group_size = 2, it will
      return the two most likely classes for each data point (in descending order
      of likelihood)
    verbose: Whether or not you want verbose classification output (for every datapoint
      instead of one for the whole sequence)
    
    Returns
    -------
    predictions: Predictions is an array of predictions represented according to 
      the original representation and an array representing the relative confidence 
      in those predictions. 
    confidence: The confidence array is matched by index to the predictions array.
    '''
    predictions = []
    confidence = []
    for i in range(X.shape[0]):
      self._predict_helper(X[i], window_size)
      pred = self.accumulated_pred
      predLen = len(pred)
      #turn the accumulated scores into a probability distribution
      report = softmax(pred.reshape(1, len(pred))).reshape(len(pred))
      #pick out the group_size largest indices
      arr = np.argsort(-pred)[:group_size]
      #generate the array of predictions/confidences for this datapoint
      prediction = []
      conf = []
      for ele in arr:
        temp = [0] * predLen
        temp[ele] = 1
        prediction.append(self.preprocessor.get_encoder().decode_label(temp))
        conf.append(report[ele])
      #append the new prediction/confidence pair to the list of predictions/confidences
      predictions.append(prediction)
      confidence.append(conf)
    
    if not verbose:
      predictions = predictions[len(predictions) - 1]
      confidence = confidence[len(confidence) - 1]

    return predictions, confidence


  def _predict_helper(self, x, window_size):
    '''
    Updates the model's state for a single datapoint. THIS IS STATEFUL.

    Parameters
    ----------
    x: A numpy vector. Expects shape(#features,)
    window_size: The size of the sequence to look at for any given prediction.
    '''
    x = x.reshape((1, x.shape[0]))
    raw_prediction = self.model.predict(x)
    #reshape into a vector
    raw_prediction = raw_prediction.reshape(raw_prediction.shape[1])
    #verify that we've accumulated 10 predictions before decrementing
    if self.last_predictions.qsize() >= window_size:
      last = self.last_predictions.get()
      self.accumulated_pred -= last
    
    self.accumulated_pred += raw_prediction
    self.last_predictions.put(raw_prediction)


def exp_decay(epoch):
    initial_lrate = 0.005
    k = 0.1
    lrate = initial_lrate * math.exp(-k*epoch)
    return lrate