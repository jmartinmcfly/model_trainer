from __future__ import division
import abc
import types
import re

# To allow for abstract static methods. 
# Found at https://stackoverflow.com/questions/4474395/staticmethod-and-abc-abstractmethod-will-it-blend/4474495#4474495

class abstractstatic(staticmethod):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True
    __isabstractmethod__ = True

import os
import curses
import fcntl
import termios
import struct
import pickle

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np

from sample.preprocessing.preprocessor import Preprocessor
from sample.preprocessing.categorical_encoder import Categorical_encoder
from sample.preprocessing.gaussian_normalizer import Gaussian_normalizer
from sample.parsing.parser import parse_json_new_model as parse_train
from sample.parsing.parser import parse_json_classify as parse_classify
import sample.utils.printer as printer

# Python magic. Unbundles a method from a static wrapper.
# To learn more, follow this 
# link: https://stackoverflow.com/questions/3083692/using-class-static-methods-as-default-parameter-values-within-methods-of-the-sam
def unstaticmethod(static):
    return static.__get__(None, object)

class Model(object):
  '''
  This class encapsulaters an abstract classification model. This file is organized
  in order of ML funnel defined in the README, which is roughly
  (train -> evaluate -> save -> load -> classify). 

  All models should be subclasses of Model. To implement this abstract class you must define:
  
  select_model:
    Trains (possibly multiple) models and selects the best one.
  save_model:
    Saves the trained model to file
  _accuracy_n:
    Calculates accuracy.
  load_saved_model:
    Loads the trained model that was saved in save_model.
  classify:
    Classifies a data point.

  You can find the I/O contracts for these methods by searching for 
  abstractmethod (select_model, save_model, _accuracy_n, classify) and 
  abstractstatic (load_saved_model).
  TODO: Make a template file for these.
  '''

  __metaclass__ = abc.ABCMeta
  MODEL_TYPE = 'abstract model'
  transformation = staticmethod(lambda x : x)
  #the model for the class
  model = None
  preprocessor = Preprocessor(encoder=Categorical_encoder(), normalizer=Gaussian_normalizer(), transformation=transformation)

  def __init__(self, model=None, preprocessor=None):
    if preprocessor == None:
      preprocessor = self.preprocessor
    #de statify the method
    transformation = preprocessor.get_transformation()
    # DO NOT TOUCH this if statement - its python magic
    # If a method held in a class is passed in, it needs to be unstaticed
    if not isinstance(transformation, types.FunctionType):
      preprocessor.transformation = unstaticmethod(transformation)
    self.preprocessor = preprocessor
    self.model = model

  '''
  SECTION: Training
  '''

  def training_pipeline(self, data, preprocessor, verbose=False, test_size=0.2,
    eval_group_size=3, eval_seq_size=5):
      '''
      Runs the entire pipeline for the given data. It parses and preprocesses the
      data, trains a model, evaluates it, and saves the relevant data to save_path.
      
      Parameters
      ----------
      data: The data to train on. Should be JSON formatted.
      preprocessor: The preprocessor object to use in this model. Defines the functionality
        for feature engineering, normalization, and encoding.
      verbose: Whether or not to supply verbose prints during training
      test_size: The proportion of the dataset to hold out for validation
      eval_group_size: The range of group sizes to evaluate
      eval_seq_size: the range of sequence sizes to evaluate. Evaluation occurs
        across the cross product of range(1, eval_group_size + 1) and
        range(1, eval_seq_size + 1)

      Returns
      -------
      An evaluation of the model: This is an array of dicts of the form 
        {'class' : class, 'accuracy' : accuracy} as well as a dict
        {'average_accuracy' : avg acc}
      A preprocessor: The same preprocessor passed in, but trained on the data.
        Holds state regarding data / label encodings, normalization terms, and
        the function used to perform feature engineering.
      '''
      # needed to get around python's weirdness around passing methods nested in
      # functions. The method has to be static to pass, and then unbound.
      # there is code in model that gets around some issues with passing static methods
      # in python
      X, Y, preprocessor = self.preprocess_train(data, preprocessor)
      #check there is enough data (necessary but not sufficient check)
      uniques, counts = np.unique(Y, axis=0, return_counts=True)
      assert np.all(counts >= 2), "You do not have at least one of each datapoint. Please collect more data."

      #TODO: Factor this out into a generator? What about k-fold validation?
      #TODO: Maybe just make a parameter kfold=False and loop if true.
      X_train, X_test, Y_train, Y_test = self.model_train_test_split(X, Y, test_size)
      # I am currently not doing anything with the training history (validation / 
      # training accuracy over epochs). It is stored in the instance variable 'history' 
      # during the execution of select_model.
      self.select_model(X_train, Y_train, verbose=verbose)
      # generate every evaluation in the cartesian product of (range(1, group_size), range(1, seq_size))
      # Each unique pair (group_size, seq_size) is mapped to its corresponding report
      evaluation = {(i, j) : self.evaluate_in_pipeline(X_test, Y_test, preprocessor.get_encoder(), i, j)
        for i in range(1, eval_group_size + 1) for j in range(1, eval_seq_size + 1)}
      return evaluation, preprocessor

  '''
  SUBSECTION: Training pipeline sub methods
  '''

  def preprocess_train(self, data, preprocessor):
    '''
    Preprocesses the data for classification. Assumes data is a JSON formatted string.

    Parameters
    ----------
    data: A JSON formatted string containing datapoint / label pairs.
    preprocessor: The preprocessor object to use in this model. Defines the functionality
      for feature engineering, normalization, and encoding.

    Returns
    -------
    X: The datapoints (Gaussian Normalized to mean=0 std_dev=1)
    Y: The labels (one hot encoded), matched to X by index
    preprocessor: Same as input but populated with the values computed during preprocessing
      (encodings, normalization constants)
    '''
    X, Y, encoder = parse_train(data, preprocessor.get_encoder())
    X = self._feature_engineering(X, preprocessor.get_transformation())
    X, normalizer = self._normalize_train(X, preprocessor.get_normalizer())
    Y, encoder = encoder.fit_labels(Y)
    X = self._feature_selection(X)
    X, Y = self._final_cleaning(X, Y)
    #reinstantiate preprocessor with new encoder and normalizer
    preprocessor = Preprocessor(encoder, normalizer, preprocessor.get_transformation())
    return X, Y, preprocessor
  
  #TODO: Should I factor this out? Maybe I shouldn't use the transformation function
  #      at all... it seems superfluous.
  def _feature_engineering(self, X, transformation):
    '''
    Takes the raw data and preprocesses the features.

    Parameters
    ----------
    X: The data, a numpy array of shape (samples, features)
    transformation: the function used to transform the data.

    Returns
    -------
    The dataset, elementwise shifted from [-99, 0] to [0, 99] (0 wraps back to 0)
    '''
    transformation = np.vectorize(transformation)
    #shift values from [-99, 0] to [0, 99]. Shifts everything (but 0s) right 100.
    X = transformation(X)
    
    return X

  def _normalize_train(self, X, normalizer):
    '''
    Normalizes the data using Gaussian normalization

    Parameters
    ----------
    X: The data, a numpy array of shape (samples, features)

    Returns
    -------
    X: the modified dataset, shifted by the average and scaled by the std dev. The 
    result is a scaled dataset w/ a mean of 0 and a std dev of 1.
    normalizer: The normalizer that has been fit to the data
    '''
    #normalize
    normalizer = normalizer.fit(X)
    #handle corner case where variance = 0 to avoid divide by 0
    X = normalizer.normalize(X)

    return X, normalizer

  def _final_cleaning(self, X, Y):
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
    # Currently the identity transformation

    return X, Y

  def _feature_selection(self, X):
    '''
    This is where dimensionality reduction would occur if desired. Currently equivalent
    to the identity matrix.

    Parameters
    ----------
    X: the engineered and normalized data

    Returns
    -------
    X with less relevant features removed (if necessary. currently no dimension reduction).
    '''
    #reducer = Preprocessor.get_dim_reduction()
    #pca = PCA(20)
    #pca.fit_transform(X)

    return X

  def model_train_test_split(self, X, Y, test_size=0.2):
    '''
    Performs a train test split specific to this model. This assumes that there
    are at least _ceiling(1 / test-size) data points per label, and will cause problems further down
    the pipeline if otherwise. For the default (test-size=0.2), you need at least 5 points of each
    class.

    Parameters
    ----------
    X: the unlabled data
    Y: the data, paired by index to X
    test_size: the size of the test set to split out

    Returns
    -------
    X_train, X_test, Y_train, Y_test. Each set (X_train, Y_train), (X_test, Y_test)
    is paired indexwise.
    '''
    # assertion to make sure we have enough of each datapoint
    if len(Y) < 300:
      uniques, counts = np.unique(Y, return_counts=True)
      assert np.all(counts >= 2), "You do not have at least one of each datapoint. Please collect more data."

    # stratify indicates that we will split on each label seperately to ensure an equal distribution
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, stratify=Y)

    return X_train, X_test, Y_train, Y_test

  @abc.abstractmethod
  def select_model(self, X, Y, verbose=False):
    '''
    Automatically finds the optimal hyperparameters and architecture for the
    NN.

    Parameters
    ----------
    X: data points
    Y: labels matched by index to the datapoints
    verbose: Whether or not to print verbose data from training. Default not verbose.

    Returns
    -------
    the trained model
    '''
    pass

  def save_pipeline_output(self, path, evaluation, preprocessor):
    '''
    Saves the output from pipeline. Will overwrite files if they exist, create
    them otherwise.

    Parameters
    ----------
    path: The path to the folder where the model's data will be saved.
    evaluation: The evaluation of the model on the held out test set
    preprocessor: The preprocessor used to preprocess the data. This state is
      needed to preprocess data for classification.

    Returns
    -------
    Evaluation: The same evaluation as was originally passed in. This is done
    to make this method more functional and retain the linear flow of I/O in the
    train script.
    '''
    path = Model.format_path(path)
    #creates the relative directory
    directory = os.path.join(path)
    #if the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    #save model
    self.save_model_type(path)
    self.save_model(path)
    #save evaluation
    with open(path + "evaluation.pkl", "wb") as f:
      pickle.dump(evaluation, f, -1)
    #save the encoding object
    with open(path + 'preprocessor.pkl', 'wb') as f:
      pickle.dump(preprocessor, f, -1)

    return evaluation

  def save_model_type(self, path):
    '''
    Saves the type of the model model at path. Is needed to call some static methods
    upon loading.

    path: The path to the parent folder to save the model at.
    '''
    with open(path + "model_type.txt", "wb") as f:
      f.write(Model.get_type(self))

  @abc.abstractmethod
  def save_model(self, path):
    '''
    Saves the model at filepath as an HDF5
    
    path: The path to the parent folder to save the model at.
    '''
    pass

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
    for i in range(len(Y[0])):
      d = X[Y[:, i] == 1]
      l = Y[Y[:, i] == 1]
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
  
  @abc.abstractmethod
  def _accuracy_n(self, X, Y, group_size=1, seq_size=1):
    '''
    Calculates the accuracy of the model where each classification is generated
    on seq_size data points and is considered correct when it the correct label
    is within the n most likely predicted categories.

    Parameters
    ----------
    X: a numpy array of inputs
      the inputs we are attempting to classify
    Y: a numpy array of labels matched by index to inputs
    group_size: the margin for error. A prediction will be considered correct
      if the correct answer is within the first group_size guesses. group_size = 1
      is equivalent to keras' normal accuracy metric.
    seq_size: The number of predictions that go into any larger prediction

    Returns
    -------
    The proportion of examples correctly classified within the first n guesses.
    Does not take into account relative confidence.
    '''
    pass

  @staticmethod
  def load_evaluation(model_type, path, group_size, seq_size, readable=False):
    '''
    Prints the evaluation for the model at path.

    Parameters
    ----------
    model_type: The python type object of the model being evaluated.
    path: The path to the model to load the evaluation from
    group_size: the group_size portion of the key tuple to the evaluation dictionary.
      Represents the margin of error you want to allow for prediction prediction success
      in the accuracy metric.
    seq_size: the seq_size portion of the key tuple to the evaluation dictionary.
      Represents how long a sequence per prediction you want accuracy metrics on.
    readable: pass readable=True if you want a human readable version printed to terminal
      in addition to returning json formatted evaluation.

    Returns
    -------
    evaluation: A dictionary of {class1 : performance, class2 : performance, ..., 
      avg_over_classes : performance, weighted_avg : performance, median_over_classes : performance}

    Raises
    ------
    ValueError: Raises this if (group_size, seq_size) is not in the evaluation dictionary.
    '''
    model = Model.get_type(model_type)
    with open(path + 'evaluation.pkl', 'rb') as f:
      evaluation = pickle.load(f)
      key = (group_size, seq_size)
      if not key in evaluation:
        raise ValueError('The report for (group_size, sequence_size) was not stored during training.')
      else:
        evaluation = evaluation[(group_size, seq_size)]
    if readable:
      printer.print_header("BEGIN REPORTS", buffer=1, bold=True, filler='=')
      header = printer.color.UNDERLINE + 'MODEL EVALUATION of the ' + model + ' at ' + path + ': ' + printer.color.RED + \
          evaluation['report_id'] + printer.color.END
      Model.print_evaluation(evaluation, header=header)
      printer.print_header("END REPORTS", buffer=1, bold=True, filler='=')
      return evaluation
    else:
      return evaluation

  @staticmethod
  def print_evaluation(evaluation, header='MODEL EVALUATION'):
    #TODO: factor evaluaton into a class
    '''
    Prints the evaluation.

    Parameters
    ----------
    evaluation: a dict of the form
      {class1_name : accuracy, class2_name : accuracy, classn_name : accuracy, avg : accuracy, median : accuracy}.
    header: The list of headers to be printed above the evaluation stats
    '''
    indent = '-- '
    print('')
    printer.print_header(header, filler='_')
    #print scores
    keys = evaluation.keys()
    scores = evaluation.values()
    #scores sorted in ascending order. Each score is represented by its corresponding class index in keys.
    scores = np.argsort(np.asarray(scores))
    printer.print_sub_header('Accuracy by Class')
    for i in scores:
      k = keys[i]
      if k != 'avg_over_classes' and k != 'median_over_classes' and k != 'weighted_avg' and k != 'weighted_median' \
        and k != 'report_id':
        v = evaluation[keys[i]]
        v = str(v)
        #the ugly str(int(float)) is to turn a
        print(indent + k + ": " + v)
    print('')
    printer.print_sub_header('Accuracy Summary')
    print(indent + 'avg_over_classes: ' + str(evaluation['avg_over_classes']))
    print(indent + 'weighted_avg: ' + str(evaluation['weighted_avg']))
    print(indent + 'median_over_classes: ' + str(evaluation['median_over_classes']))
    print('')
    printer.print_line_break(break_char='_ ')

  '''
  SECTION: Classification
  '''

  @staticmethod
  def load_classification_pipeline(model_type, path):
    '''
    Loads the necessary files to execute the classification pipeline

    Parameters
    ----------
    model_type: The type of the model being loaded.
    path: The path to the folder housing the model

    Returns
    -------
    model_raw: The saved model
    Preprocessor: The preprocessor that was used to train the saved model

    Raises
    ------
    ValueError, ImportError
    '''
    path = Model.format_path(path)
    #load up the model and other necessary files
    try:
      return model_type.load_saved_model(path)
    except (ValueError, ImportError):
      raise
  
  @staticmethod
  def exec_classification_pipeline(data, model, group_size=1):
    '''
    Takes in a to a model and data, loads the model at path, and then delivers 
    a single prediction on the data consisting of the model's group_size best guesses.

    Parameters
    ----------
    data: the data to classify. Expects a JSON formatted string.
    model: The model to classify with. Should be fully loaded (aka the output
      of load_classification_pipeline).
    group_size: the number of guesses to include in the prediction

    Returns
    -------
    A single classification for the sequence. The classification is an array of dictionaries of
    the form {'prediction' : class, 'confidence' : probability} in descending order
    of confidence.
    '''
    #parse the data
    preprocessor = model.preprocessor
    X = parse_classify(data, preprocessor.get_encoder())
    X = model.preprocess_classify(X, preprocessor.get_normalizer(), 
      preprocessor.get_transformation())
    predictions, confidence = model.classify(X, len(data), group_size)
    classification = [{'prediction' : p, 'confidence' : c} for p, c in zip(predictions, confidence)]
    return classification

  def preprocess_classify(self, X, normalizer, transformation):
    '''
    Preprocesses the data for classification. Assumes data is a JSON formatted string.

    Parameters
    ----------
    X: The input to be preprocessed for classification. A JSON formatted string.
    normalizer: The normalizer used to normalize the data during the training
      of the model so that the model is classifying on a consistent 'universe'.
    transformation: The function used to perform elementwise transformation on the data
      during training so that the model is classifying on a consistent 'universe'.

    Returns
    -------
    A numpy array of shape (samples, features).
    '''
    X = self._feature_engineering(X, transformation)
    X = self._normalize_classify(X, normalizer)
    return X

  def _normalize_classify(self, X, normalizer):
    '''
    Parameters
    ---------
    X: data
    normalizer: The normalizer that was used during the training of the model

    Returns
    -------
    normalized: X engineered and normalized in a way that (given the correct params) is equivalent
        to the preprocessing performed on the training data
    '''
    #normalize
    normalized = normalizer.normalize(X)
    return normalized

  @abstractstatic
  def load_saved_model(path):
    '''
    Loads the model of the given name at the path.

    Parameters
    ----------
    path: a path to the folder holding the model. Should be of the form
      'path_to_all_models/model_name'
    '''
    with open(path + 'preprocessor.pkl', 'rb') as f:
      preprocessor = pickle.load(f)
    return preprocessor

  @abc.abstractmethod
  def classify(self, X, group_size=1):
    '''
    Predicts the group_size most likely rooms given a set of readings. Assumes
    that the classes are represented as integers.

    Parameters
    ----------
    X: A numpy matrix of vectors of wifi signal strenghts.
    group_size: The number of rooms to return for each reading. If group_size = 2, it will
      return the two most likely rooms for each data point (in descending order
      of likelihood). Default is 1.
    
    Returns
    -------
    (Predictions, Confidence): Predictions is an array of predictions represented 
      according to the encoding and an array representing the relative confidence 
      in those predictions. The confidence array is matched by index to the predictions 
      array.
    '''
    pass

  @staticmethod
  def print_classification(classification):
    #TODO: Extend classification
    '''
    Prints a classification of the form [{pred1 : conf}, {pred2 : conf}, ...]

    Parameters
    ----------
    classification: a classification of the form [{pred1 : conf}, {pred2 : conf}, ...]
    '''
    print('')
    header = 'SEQUENCE CLASSIFICATION'
    printer.print_header(header)
    for i, guess in enumerate(classification):
      k = guess['prediction']
      v = guess['confidence']
      print("Prediction " + str(i + 1) + ": ")
      print("-- " + "class: " + str(k))
      print("-- " + "confidence: " + str(v))
      if i < len(classification) - 1:
        print
    printer.print_line_break()

  @staticmethod
  def format_path(path):
    '''
    makes sure path ends with '/'

    Parameters
    ----------
    path: a path to check
    '''
    if path[-1] != '/':
      path += ('/')
    return path

  '''
  These methods are called to help determine the subtype of the model in the scripts.
  '''

  @staticmethod
  def get_type(model_name):
    '''
    Returns the name of the model_type in string form.
    '''
    return model_name.MODEL_TYPE

  @staticmethod
  def load_type(path):
    '''
    Returns the type of the model saved at path.
    '''
    try:
      return open(path + 'model_type.txt').read()
    except IOError:
      raise

  