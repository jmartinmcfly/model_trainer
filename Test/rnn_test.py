import pytest
import os
from sample.models.rnn import Rnn, model_train_test_split
import numpy as np
import json

def test_train_test_split():
  data = np.asarray([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
    [13, 14, 15],
    [16, 17, 18]
  ])
  labels = np.asarray([
    [0, 1],
    [1, 0],
    [1, 0],
    [1, 0],
    [0, 1],
    [1, 0]
  ])
  x_train, x_test, y_train, y_test = model_train_test_split(data, labels, .3)
  assert np.all(x_train == np.asarray([[1, 2, 3], [13, 14, 15], [7, 8, 9], [10, 11, 12], [16, 17, 18]]))
  assert np.all(x_test == np.asarray([[4, 5, 6]]))
  assert np.all(y_train == np.asarray([[0, 1], [0, 1], [1,0], [1, 0], [1, 0]]))
  assert np.all(y_test == np.asarray([[1, 0]]))
  data = np.asarray([
    [1, 2, 3]
  ])
  labels = np.asarray([
    [1]
  ])
  x_train, x_test, y_train, y_test = model_train_test_split(data, labels, .3)
  assert np.all(x_train == np.asarray([[1, 2, 3]]))
  assert np.all(y_train == np.asarray([[1]]))
  assert x_test.size == 0
  assert y_test.size == 0

def test_accuracy_n():
  pass

def test_classify():
  pass

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
