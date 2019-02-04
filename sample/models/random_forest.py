from __future__ import division

import sys
import Queue as q
import pickle
import os

import dill
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.utils.extmath import softmax
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from sample.parsing.parser import parse_json_new_model as parse_train
from sample.parsing.parser import parse_json_classify as parse_classify
from sample.preprocessing.preprocessor import Preprocessor
from sample.preprocessing.gaussian_normalizer import Gaussian_normalizer
from sample.preprocessing.categorical_encoder import Categorical_encoder
from sample.preprocessing.rf_encoder import Rf_encoder
from .model import Model, unstaticmethod

'''
The file is organized preprocessing -> training -> evaluation -> classification. 
All non stateful helper methods (of the form _helper_name) are placed at the
end of the file, outside of the scope of Dnn.
'''

class Random_forest(Model):
  '''
  This class encapsulaters a model (a Keras Neural Net) to classify which room
  a user is in using wifi signal strengths for a single floor.
  '''
  #the model type to be referred to in scripts
  MODEL_TYPE = 'random_forest'
  #the history (training loss and validation loss by epoch) from training of the model
  history = None
  #the accumulated prediction
  accumulated_pred = None
  #the last predictions stored as a queue - you decrement accumulated by the
  #last seen prediction
  last_predictions = q.Queue()
  #the function for transforming wifi values
  transformation = staticmethod(lambda x : (x + 100) % 100)
  preprocessor = Preprocessor(encoder=Rf_encoder(), normalizer=Gaussian_normalizer(),
    transformation=transformation)
  #training params
  epochs = 300

  def __init__(self, model=None, preprocessor=None):
    if preprocessor == None:
      preprocessor = self.preprocessor
    super(Random_forest, self).__init__(model=model, preprocessor=preprocessor)
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
    X, indices = np.unique(X, axis=0, return_index=True)
    Y = Y[indices]

    return X, Y

  def select_model(self, X, Y, verbose=False, validation_size=0.2):
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
    num_labels = np.max(Y)
    print(Y)
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
    clf.fit(X, Y)
    print(clf.predict_proba([X[0]]))
    print(clf.predict([X[0]]))
    self.model = clf

  def save_model(self, filepath):
    '''
    Saves the model at filepath as an HDF5. Expects filepath to end in .h5.

    Parameters
    ----------
    filepath: A filepath
    '''
    with open(filepath + 'model.pkl', 'wb') as f:
      pickle.dump(self.model, f)

  '''
  SECTION: Evaluation
  '''

  def evaluate_in_pipeline(self, X, Y, encoder, group_size=1, seq_size=1):
    '''
    Evaluates self.model on the given data.

    Parameters
    ----------
    X: the inputs
    Y: the labels, matched to X by index
    encoder: An encoder that can decode labels to their original representation
    group_size: The amount of slack for a correct prediction. If the correct guess
      is within the first group_size guesses, than a prediction is considered correct.
    seq_size: The amount of datapoints to consider in every prediction.

    Returns
    -------
    A dictionary matching class to accuracy. It also has an "average_accuracy" field 
    which maps to the average accuracy over all datapoints (note this is different than all classes).
    '''
    #sort the labels into batches
    partitioned_X = []
    partitioned_Y = []
    # loop over the different labels. This assumes one hot encoding
    for i in range(np.max(Y)):
      d = X[Y[i] == 1]
      l = Y[Y[i] == 1]
      # just in case the test data is missing a label
      if d.size is not 0:
        partitioned_X.append(d)
        partitioned_Y.append(l)
    #collect accuracy stat for each label
    accuracy = {}
    accuracy_list = np.zeros(len(partitioned_Y))
    weight_list = np.zeros(len(partitioned_Y))
    # Loop over each class and generate accuracy metrics
    for i in range(len(partitioned_Y)):
      # grab the label
      val = partitioned_Y[i][0]
      # TODO: FIX VAL HERE
      print(val)
      # decode the label to its value
      label = encoder.decode_label(val)
      # label may be a float but we want an int in string form
      stringified_label = str(int(label))
      # Map the label name to its accuracy
      accuracy[stringified_label] = self._accuracy_n(partitioned_X[i], partitioned_Y[i], 
        group_size=group_size, seq_size=seq_size)
      #add the weighted accuracy to total_accuracy
      accuracy_list[i] = accuracy[stringified_label]
      #weight by proportion of dataset
      weight_list[i] = len(partitioned_X[i]) / len(X)

    #add the weighted average of accuracy over each class to the dict
    accuracy['avg_over_classes'] = np.mean(accuracy_list)
    accuracy['weighted_avg'] = np.sum(accuracy_list * weight_list)
    accuracy['median_over_classes'] = np.median(accuracy_list)
    accuracy['report_id'] = 'GROUP SIZE = ' + str(group_size) + ', SEQUENCE SIZE = ' + str(seq_size)

    return accuracy

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
    preprocessor = super(Random_forest, Random_forest).load_saved_model(path)

    try:
      with open(path + 'model.pkl', 'rb') as f:
        model = pickle.load(f)
    except ValueError:
      raise ValueError("The model savefile is invalid.")
    
    #TODO: Consider factoring out loading/saving logic into encoder/normalizer class
    #      Only issue is with loading logic... need to know which subclass of encoder
    #      and normalizer we are dealing with
    
    return Random_forest(model, preprocessor)

  def classify(self, X, window_size=10, group_size=1, verbose=False):
    #TODO: Factor predictions into a prediction class, with two entries - prediction, and confidence
    '''
    Predicts the group_size most likely rooms given a set of readings. Assumes
    that the classes are represented as integers.

    Parameters
    ----------
    X: A numpy matrix of vectors of wifi signal strengths. Expects shape (#samples,#features)
    window_size: The number of readings to take into account for the final reading.
    group_size: The number of rooms to return for each reading. If group_size = 2, it will
      return the two most likely rooms for each data point (in descending order
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
    # No fancy accum, just basics
    predictions = []
    confidence = []
    print(self.model.n_classes_)
    pred = self.model.predict_proba(X)
    print(pred)
    for p in pred:
      predLen = len(p)
      print(p)
      #pick out the group_size largest indices
      arr = np.argsort(-p)[:group_size]
      #generate the array of predictions/confidences for this datapoint
      prediction = []
      conf = []
      for ele in arr:
        temp = [0] * predLen
        print(ele)
        temp[ele] = 1
        prediction.append(self.preprocessor.get_encoder().decode_label(temp))
        conf.append(p[ele])
      #append the new prediction/confidence pair to the list of predictions/confidences
      predictions.append(prediction)
      confidence.append(conf)
    
    if not verbose:
      predictions = predictions[len(predictions) - 1]
      confidence = confidence[len(confidence) - 1]

    return predictions, confidence

    

