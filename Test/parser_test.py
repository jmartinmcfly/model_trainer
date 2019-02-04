import pytest
import os
from sample.parsing.parser import feature_to_col, parse_json_new_model, _encode_existing
from sample.preprocessing.categorical_encoder import Categorical_encoder
import numpy as np
import json

def test_feature_dict():
  assert feature_to_col([]) == {}
  assert feature_to_col(
      [{"1234": '-92', "class": "nothing"}]) == {"1234": 0}
  assert feature_to_col([{"12:34:56": '-92', "class": "nothing"},
    {"1234" : '-92', "class" : "nothing"}]) == {"12:34:56" : 0, "1234": 1}
  assert feature_to_col([{"1234": '-92', "class": "nothing"},
    {"1234" : '-92', "class" : "nothing"}]) == \
    {"1234" : 0}


@pytest.mark.parametrize("file", [
  ("data/iris.json")])

def test_parse_json_new_model(file):
  data = open(file).read()
  dataset, labels, encoder =  parse_json_new_model(data, Categorical_encoder())
  #check that the shape feature_to_index, dataset, coded_labels, and encoding_dict
  #all match
  assert dataset.shape[1] == len(encoder.data_encoder.keys())
  assert dataset.shape[0] == labels.shape[0]
  
def _encode_existing_test():
  enc_dict = {"a" : [1, 0, 0], "b" : [0, 0, 1], "c" : [0, 1, 0]}
  encoded = _encode_existing(["a", "a", "c", "b"], enc_dict)
  assert np.array_equal(encoded[0], [1, 0, 0])
  assert np.array_equal(encoded[1], [1, 0, 0])
  assert np.array_equal(encoded[2], [0, 1, 0])
  assert np.array_equal(encoded[3], [0, 0, 1])
