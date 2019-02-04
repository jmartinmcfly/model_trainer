import sys
import argparse
import json
import os

from sample.models.model import Model
from sample.models.model_list import get_models
import sample.utils.printer as printer

'''
This script allows the training of any model that correctly subclasses Model. It
train, evaluates, and stores the model. Currently, it uses the default transform_function (used in preprocessing)
of each class.
'''

# TODO: I need to figure out how to allow the user to pass a function. But this is hard over cmd line...

#All of the available models to train
models = get_models()

parser = argparse.ArgumentParser()

parser.add_argument('model_type', help='The type of model to train.')
parser.add_argument('data', help='The data to train on. Either a JSON formatted string or a path to a file. ' +
  'If it is a file, use the --isfile flag.')
parser.add_argument('models_folder', help='The parent folder to place the folder containing the new model in')
parser.add_argument('model_name', help='The name of the folder to store the new model to. If the folder already ' +
  'exists, its existing contents will be overriden. If it does not, it will be created.')
parser.add_argument('group_size', help='OPTIONAL ARGUMENT. The first of a pair that dictates the evaluation that will be produced ' +
  'by the training when it is finished. The second is seq_size.' + 'This will generate a report for every element of the cartesian ' +
  'product of the sets range(group_size) and ' + 'range(sequence_size). Each pair (group_size, sequence_size) ' + 
  'generates a full report where each prediction is. ' + 'generated over sequence_size data points and is ' +
  'considered correct if it has a correct prediction within ' + 'the first group_size guesses ' +
  '(ordered by likelihood). Note that unless you held out data you will be unable to ' +
  'evaluate the model on the training data after this script is finished running, so generous evaluation ' +
  'parameters are recommended. Also note that this may not be supported for every model type.', 
  nargs='?', type=int, default=1)
parser.add_argument('seq_size', help='OPTIONAL ARGUMENT. Must also supply a group size if you want to supply a sequence size. ' +
  'See help message for group_size.', nargs='?', type=int, default=10)
parser.add_argument('--isfile', help='Use this flag when data is a path to a JSON formatted file.', 
  action='store_true')
parser.add_argument('--display', help='This flag will show you all possible models.', 
  action='store_true')
parser.add_argument('--verbose_train', help='Pass this flag if you want verbose training data.', 
  action='store_true')
parser.add_argument('--v_eval', help='Pass this flag if you want evaluation data printed. ' + 
  'Either way it will be saved.', action='store_true')
parser.add_argument('--vv_eval', help='Pass this flag if you want all evaluation data printed. ' + 
  'Either way it will be saved.', action='store_true')
parser.add_argument('--vv', help='Pass this flags to set all verbose flags as true', action='store_true')


# Parse arguments
# ---------------
args = parser.parse_args()

if args.vv:
  args.verbose_train = True
  args.vv_eval = True

# Parse model_type: instantiate a model of the user defined type to train
# Lower case the passed model_type to minimize invalid model type errors
model_type = args.model_type.lower()
# If the user passed an invalid model type throw an error and exit
if not model_type in models.keys():
  sys.stdout.write('You passed on invalid model type. These are the available models: \n')
  model_names = models.keys()
  model_names.sort()
  for k in model_names:
    sys.stdout.write('-- ' + k + '\n')
  sys.stdout.flush()
  sys.exit(1)
model = models[model_type]
m = model()
'''
If you want to pass a custom transform function (as you may want to do with 
future (aka non wifi) datasets, uncomment this code.

transform_features = //NOTE: INSERT DESIRED LAMBDA OR FUNCTION HERE
m = model(transform_features=transform_features)
'''

# Parse eval_params:
group_size = args.group_size
seq_size = args.seq_size

# Parse data:
data = args.data

# Parse the save path:
path = args.models_folder + '/' + args.model_name + '/'

# Parse display flag:
if not model_type in models.keys():
  sys.stdout.write('These are the available models: \n')
  model_names = models.keys()
  the_models.sort()
  for k in model_names:
    sys.stdout.write('-- ' + k + '\n')
  sys.stdout.flush()
  sys.exit(1)

# Parse isfile flag:
# if the user passed the isfile flag then data is a filepath and must be 
# loaded an read
if args.isfile:
  data = open(data).read()

# create the all missing elements of the passed path if they do not exist
if not os.path.exists(path):
    os.makedirs(path)

#This trains the model with training_pipeline and then passes the outputs directly to save_pipeline
if args.verbose_train:
  evaluation = m.save_pipeline_output(path, *m.training_pipeline(data, m.preprocessor, verbose=True,
    eval_group_size=group_size, eval_seq_size=seq_size))
else:
  evaluation = m.save_pipeline_output(path, *m.training_pipeline(data, m.preprocessor, verbose=False,
  eval_group_size=group_size, eval_seq_size=seq_size))

if args.vv_eval:
  #print evaluation info
  print('')
  printer.print_header("BEGIN REPORTS", buffer=1, bold=True, filler='=')
  # Sort the keys so that the prints are ordered for g in group_size: for j in seq_size: print
  keys = sorted(evaluation.keys())
  keys.reverse()
  for k in keys:
    header = printer.color.UNDERLINE + "MODEL EVALUATION: " + printer.color.RED + \
      evaluation[k]['report_id'] + printer.color.END
    Model.print_evaluation(evaluation[k], header=header)
  printer.print_header("END REPORTS", buffer=1, bold=True, filler='=')
elif args.v_eval:
  # just print the evaluation for (group_size, seq_size) = (1, 1)
  Model.print_evaluation(evaluation[(1,1)])
else:
  sys.exit(0)


