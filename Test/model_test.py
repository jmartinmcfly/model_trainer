import pytest
import json
import re
import shutil

from sample.models.dnn import Dnn
from sample.models.rnn import Rnn
from sample.models.model import Model
from sample.preprocessing.preprocessor import Preprocessor

from sample.preprocessing.gaussian_normalizer import Gaussian_normalizer
from sample.preprocessing.categorical_encoder import Categorical_encoder
from sample.parsing.parser import parse_json_classify as parse_classify
from sample.models.model_list import get_models

import numpy as np

models_to_test = get_models().values()


@pytest.mark.timeout(5)
@pytest.mark.parametrize("data", [
    ("data/iris.json")])

def test_models_pipeline_with_read_write(data):
  '''
  Tests the exact calls used in train and classify in order, including file writes
  and reads. Deletes test files when done.
  
  Verifies that the outputs of save_pipeline_output line up with load_classification_pipeline,
  that evaluation is formatted correctly (print won't work otherwise), that classification 
  meets a few basic structural criteria, and generally that the pipeline runs through 
  without crashing.
  '''
  path = 'test_models/test_models_pipeline_with_read_write'
  data = open(data).read()
  for model in models_to_test:
    m = model()
    print(Model.get_type(m))
    # run training
    outputs = m.training_pipeline(data, m.preprocessor, test_size=0.2)
    evaluation = outputs[0].values()[0]
    # weakly checks evaluation's structure by checking compatability with print_evaluation
    Model.print_evaluation(evaluation)
    assert 'avg_over_classes' in evaluation
    assert 'weighted_avg' in evaluation
    assert 'median_over_classes' in evaluation
    #assert that evaluation contains stats on at least two classes in addition
    #to summary data
    assert len(evaluation.keys()) >= 4
    m.save_pipeline_output(path, *outputs)
    loaded = Model.load_classification_pipeline(model, path)
    #loop group size 1, 2
    for i in range(1, 3):
      classification = Model.exec_classification_pipeline(data, loaded, group_size=i)
      # assert that there are group_size predictions within classification
      assert len(classification) == i
      #loop over predictions within classification
      for j, guess in enumerate(classification):
        # assert that confidence is a valid probability
        assert guess['confidence'] > -0.000001 and guess['confidence'] < 1.0000001
        # assert that prediction is in the dict, and not as None
        assert guess['prediction'] is not None
        # assert that the dict is in descending order of confidence
        if j != 0:
          assert classification[j - 1]['confidence'] >= classification[j]['confidence']
    
    #delete saved files
    shutil.rmtree(path)


@pytest.mark.timeout(5)
@pytest.mark.parametrize("data", [
    ("data/iris.json")])

def test_models_pipeline_methods(data):
  '''
  Tests the model pipeline on all available models. Promises very basic things:
  that the pipeline runs without crashing, and that certain basic properties
  of I/O are fulfilled, and that the outputs from train (if saved and loaded succesfully),
  are the proper inputs to classify.

  This does not guarantee the functionality of the train_pipeline or the 
  classification_pipeline (those also save to and load from files). However, it
  does test their basic logic without the file I/O.
  '''
  f = data
  data = open(data).read()
  for model in models_to_test:
    #test that parse works
    preprocessor = Preprocessor()
    m = model(preprocessor=preprocessor)
    print(Model.get_type(m))
    X, Y, preprocessor = m.preprocess_train(data, preprocessor)
    m.preprocessor = preprocessor
    encoder = preprocessor.get_encoder()
    normalizer = preprocessor.get_normalizer()
    X_train, X_test, Y_train, Y_test = m.model_train_test_split(X, Y, test_size=0.2)
    #assert that every sample has a label
    assert len(X_train) == len(Y_train)
    assert len(X_test) == len(Y_test)
    x_train_unique = np.unique(X_train, axis=0)
    x_test_unique = np.unique(X_test, axis=0)

    combined = np.vstack((x_train_unique, x_test_unique))
    #assert that test and train are disjoint
    assert len(np.unique(combined, axis=0)) == (len(x_train_unique) + len(x_test_unique))
    #test that training works
    m = model()
    metadata = m.select_model(X_train, Y_train)
    #test that evaluate works
    evaluation = m.evaluate_in_pipeline(X_test, Y_test, encoder)
    #sanity check on accuracy value
    assert evaluation['avg_over_classes'] >= -0.0000001 and evaluation['avg_over_classes'] <= 1.00000001
    #assert that evaluation contains accuracy for each class and an average_accuracy
    #NOTE: This is the number of none integer / float strings, and only is accurate if
    #      all classes are represented by ints/floats.
    excess_keys = len([k for k in evaluation.keys() if re.match("\d+(\.\d+?)?", k) is None])
    assert len(evaluation) == (len(np.unique(Y_test, axis=0)) + excess_keys)
    #I will choose not to test save because testing for the existence of files is
    #annoying, and this problem will be fairly obvious if it occurs

    #tests the whole classification pipeline for basic properties
    #handles everything but loading from file.
    X = parse_classify(data, encoder)
    X = m.preprocess_classify(X, normalizer, m.transformation)
    m = model(m.model, preprocessor)
    predictions, confidence = m.classify(X, window_size=len(data), group_size=1)
    classification = [{'prediction' : p, 'confidence' : c} for p, c in zip(predictions, confidence)]
    assert len(classification) == 1
    assert len(classification[0]) == 2
    assert 'prediction' in classification[0]
    assert 'confidence' in classification[0]
    predictions, confidence = m.classify(X, len(data), 2)
    classification = [{'prediction' : p, 'confidence' : c} for p, c in zip(predictions, confidence)]
    assert len(classification) == 2
    assert len(classification[0]) == 2
    assert len(classification[1]) == 2
    assert 'prediction' in classification[0] and 'prediction' in classification[1]
    assert 'confidence' in classification[0] and 'confidence' in classification[1]
    assert classification[0]['confidence'] > classification[1]['confidence']