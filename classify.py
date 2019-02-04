import argparse
import sys
import json

from sample.models.model import Model
from sample.models.model_list import get_models

'''
The script runs the classifier on the given data and returns a single classification -
the room that the model thinks the person is in.

Parameters
----------
data: the json formatted data to classify
models_folder: The parent folder containing the models
model_name: The folder containing the model
group_size: The number of guesses to return (ordered most to least likely)

Return
------
predictions: A single JSON array representing the group_size (default 1) most likely predictions 
  for the location of the sequence. Each object is of the schema 
  {prediction : class, confidence : probability}. The array is structured in descending order 
  of likelihood. If the --verbose flag is passed, then the script returns a two dimensional 
  JSON array, with each row structured as described above. Each row represents the prediction
  on the sequence up to the corresponding datapoint.
'''

parser = argparse.ArgumentParser(description='Classify the data given a model. The default behavior is to ' +
  'consider the data passed as a single sequence which receives a single classification.')

parser.add_argument('data', help='The data to classify. By default, it expects a JSON string. ' +
  'If you passed a filepath, please use the --isfile flag.')
parser.add_argument('path_to_models', help='The path to the folder housing all of the models.')
parser.add_argument('model_name', help='The name of the folder holding the model.')
#NOTE: not yet implemented
parser.add_argument('group_size', help='The number of guesses to return. Optional.', type=int, nargs='?', default=1)
parser.add_argument('--independent', help='Provides a prediction on every samples, treating them ' +
  'as individual, unconnected readings (no accumulation).',  action='store_true')
#NOTE: not yet implemented
parser.add_argument('--verbose', help='Still treats the predictions as a sequence (with a moving window of '
  'size 20), put returns a prediction for each point along the way.', action='store_true')
parser.add_argument('--readable', help='Pass if you want human readable output printed to terminal ' +
  'instead of a JSON string output to stdout.', action='store_true')
parser.add_argument('--isfile', help='If data is a path to a JSON file instead of '
  + 'a formatted string, use this flag. Used for testing for the terminal.',
  action='store_true')
parser.add_argument('--persistant', help='Use this to keep a session live until it receives ' +
  'a STOP signal', action='store_true')

args = parser.parse_args()

path = args.path_to_models + '/' + args.model_name + '/'
data = args.data
group_size = args.group_size

#load file at data if it is a path
if args.isfile:
  try:
    data = open(data)
  except OSError:
    sys.exit("Error opening the file. Double check you passed a valid filepath. If you " +
      "are attempting to pass a JSON formatted string, don't use the --isfile flag.")
  else:
    try:
      data = data.read()
    except IOError:
      sys.exit("Success opening the file, but error in reading. Check that the file is not corrupted.")

if args.independent:
  pass
elif args.verbose:
  pass
else:
  model_type = get_models()[Model.load_type(path)]
  #a tuple containing all info need to classify
  loaded_model = Model.load_classification_pipeline(model_type, path)
  classification = Model.exec_classification_pipeline(data, loaded_model, group_size)

if args.readable:
  Model.print_classification(classification)
else:
  sys.stdout.write(json.dumps(classification))
  sys.stdout.flush()
  sys.exit(0)