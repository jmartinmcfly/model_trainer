import argparse
import json
import sys

from sample.models.model import Model
from sample.models.dnn import Dnn
from sample.models.model_list import get_models
from sample.parsing.parser import parse_json_accuracy as parse_accuracy
import sample.utils.printer as printer

parser = argparse.ArgumentParser()

parser.add_argument('models_path', help='Path to parent folder of the model')
parser.add_argument('model_name', help='Name of the folder containing all the model data (within models_path)')
parser.add_argument('dataToTest', help='The data to evaluate the model on')
parser.add_argument('evalGroupSize', help='Will evaluate up to this group_size',
  nargs='?', type=int, default='3')
parser.add_argument('evalSeqSize', help='Will evaluate up to this sequence_size',
  nargs='?', type=int, default='3')
parser.add_argument('--readable', help='Pass this flag if you want human readable output printed to terminal',
  action='store_true')
parser.add_argument('--isfile', help='Pass this flag if you are passing the data as a path to a json file',
action='store_true')

args = parser.parse_args()

path = args.models_path + '/' + args.model_name + '/'
model_type = get_models()[Model.load_type(path)]

if args.isfile:
  data = open(args.dataToTest).read()

model_type = get_models()[Model.load_type(path)]
#a tuple containing all info need to classify
loaded_model = Model.load_classification_pipeline(model_type, path)
preprocessor = loaded_model.preprocessor
X, Y = parse_accuracy(data, preprocessor.encoder, preprocessor.normalizer)
eval_group_size = args.evalGroupSize
eval_seq_size = args.evalSeqSize
evaluation = {(i, j) : loaded_model.evaluate_in_pipeline(X, Y, preprocessor.get_encoder(), i, j)
        for i in range(1, eval_group_size + 1) for j in range(1, eval_seq_size + 1)}
if args.readable:
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
else:
  print(evaluation)