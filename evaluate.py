import argparse
import json
import sys

from sample.models.model import Model
from sample.models.model_list import get_models

parser = argparse.ArgumentParser()

parser.add_argument('models_path', help='Path to parent folder of the model')
parser.add_argument('model_name', help='Name of the folder containing all the model data (within models_path)')
parser.add_argument('group_size', help='The group_size in the tuple (group_size, seq_size) eval to load. ' +
  'Must be an integer.', type=int)
parser.add_argument('seq_size', help='The seq_size in the tuple (group_size, seq_size) eval to load. ' +
  'Must be an integer.', type=int)
parser.add_argument('--readable', help='Pass this flag if you want human readable output printed to terminal',
  action='store_true')


args = parser.parse_args()

path = args.models_path + '/' + args.model_name + '/'
model_type = get_models()[Model.load_type(path)]

# Parse eval_params:
group_size = args.group_size
seq_size = args.seq_size

if args.readable:
  print('')
  Model.load_evaluation(model_type, path, group_size, seq_size, readable=True)
  sys.exit(0)
else:
  evaluation = Model.load_evaluation(model_type, path, group_size, seq_size, readable=False)
  sys.stdout.write(json.dumps(evaluation))
  sys.stdout.flush()
  sys.exit(0)