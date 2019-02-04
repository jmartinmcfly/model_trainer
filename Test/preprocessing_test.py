import numpy as np

from sample.preprocessing.categorical_encoder import Categorical_encoder

def test_categorical_label_encoding():
  encoder = Categorical_encoder()
  labels, encoder = encoder.fit_labels([])
  output = (labels, encoder.label_encoder)
  assert output == ([], {})

  output = encoder.fit_labels(["a"])
  coded_labels = output[0]
  encoding = output[1].label_encoder
  decoding = output[1].label_decoder
  assert (coded_labels, encoding, decoding) == ([[1]], {"a" : [[1]]}, {0 : "a"})

  outputs = encoder.fit_labels(["a", "b"])
  values = (outputs[0], outputs[1].label_encoder)
  assert np.array_equal(values[0], [[1, 0], [0, 1]])
  assert np.array_equal(values[1]["a"], [1, 0])
  assert np.array_equal(values[1]["b"], [0, 1])
  assert len(values[1].keys()) == 2

  outputs = encoder.fit_labels(["a", "b", "a"])
  values = (outputs[0], outputs[1].label_encoder)
  assert np.array_equal(values[0], [[1, 0], [0, 1], [1, 0]])
  assert np.array_equal(values[1]["a"], [1, 0])
  assert np.array_equal(values[1]["b"], [0, 1])
  assert len(values[1].keys()) == 2

  outputs = encoder.fit_labels(["a", "a", "a"])
  values = (outputs[0], outputs[1].label_encoder)
  assert np.array_equal(values[0], [[1], [1], [1]])
  assert np.array_equal(values[1]["a"], [1])
  assert len(values[1].keys()) == 1