# ML Pipeline

## Introduction

This repo contains all code needed to train a dense feed forward neural and recurrent neural network, as well as most of the boilerplate skeleton required to *train any machine learning model*. The current code **only supports classification and therefore does not support regression**. However, regression can be easily implemented with a new select_model method and a new evaluation method. Also, the random_forest is slightly buggy and should only be used to determine a baseline for your dataset (not to classify). Note that the bug with random_forest currently causes "pytest" to fail. Therefore, I have commented out random_forest in the list of models that resides in sample/models/model_list.py. If you want to use the random_forest, go to sample/models/model_list.py and uncomment the line about random_forest.

A note about dataset size: the current implementation may fail on incredibly small datasets (especially the rnn) due to issues with train_test_split (all models) or issues with TimeSeriesGenerator (rnn specific). These limitations are aligned with an intuitive constraint - **you need a decent amount of datapoints per class (20+) before classification machine learning becomes viable in even the weakest sense**. If you *really* want to train on small datasets, try tweaking the test_size in train_test_split (note that an even split is the most generous possible split in this regard, as it gives you the maximize floor in regards to the size of the datasets that we need to work with). You should also note that the default form of validation employed by the current implementation is a basic cross-validation - this works well for large datasets, but can grossly misestimate accuracy on very small datasets. In the case that you want to work with a very small dataset, I would recommend attempting to implement k-fold validation. This should be done in the select_model portion of the funnel.

## Dependencies

Note 1: for those working on a google cloud vm: you need to use sudo / switch to
root permissions to use pip install.

Note 2: Matplotlib can sometimes cause problems (it is not needed for any core functionality, don't worry.) on install (which would show up as an error when you run "pip install -r requirement.txt"). 
This shouldn't affect your other installs. You will probably be missing non-python specific packages,
see [here](https://stackoverflow.com/questions/31498495/error-when-installing-using-pip). 
On Ubuntu, the command "sudo apt-get install libtiff5-dev libjpeg8-dev zlib1g-dev libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python-tk" may help.

Create a new virtual environment with the command "mkvirtualenv my-new-project".
Note that this also activates the environment. To deactivate it, use "deactivate".
You can switch to / activate any virtualenv using the command "workon env_name".

**Installation instructions on a new machine:**

1. First verify that you have pip installed (it should come with most python installs, ie Anaconda).
2. (Optional Step) Next, I would recommend installing all of your dependencies in a virtual environment: [Tutorial here](https://realpython.com/python-virtual-environments-a-primer/)
    * First, we will install a package that helps us manage virtual environments
      * pip install virtualenv
      * pip install virtualenvwrapper
      * You now have to set up the bash script for virtualenvwrapper.
        * Use the command: "which virtualenvwrapper.sh", then copy down the path. Next,
        * open  ~/.bashrc and add the line "source [path]". Finally, to reload the startup file, run "source ~/.bashrc".
    * Now, create a new virtual environment for this project. Type "mkvirtualenv my_venv_name". This will also switch you to the new virtual environment. To deactivate the environment, run "deactivate". To activate a virtual environment once it's been created, or switch from one virtual environment to another, run "workon my_venv_name".
3. run "pip install -r requirements.txt". If you intend to use a virtual environment, make sure it's activated before you install!
4. Congrats! You're ready to run everything in the repo.

## Usage

The repo houses three python scripts - one to train and save a model, one to evaluate a model, and one to load a model and classify some data (train.py, evaluate.py, and classify.py respectively).

train.py usage:

    Description: Trains a model of a given type on the data passed and saves all relevant data to the supplied path (created from path_to_parent_folder + '/' + model_name). If the path already exists, the files within model_name will be overwritten. If any of the folders in the path do not exists, the rest of the path will be created from that point. NOTE: THIS MEANS THAT YOU MAY SAVE TO A TYPO AND THE SCRIPT WILL NOT COMPLAIN SO BE CAREFUL ABOUT WHICH SAVE PATH YOU PASS.

    train.py model_type data path_to_parent_folder model_name [--isfile] [--verbose_train] [--verbose_eval] [--vv] [--display]

    Mandatory Arguments (given in order):
      model_type: The type of model to train (ie 'dnn', 'rnn', 'random_forest')
      data_to_train_on: The data to train on. Should be a JSON formatted string or
        a file housing a JSON formatted string.
      path_to_parent_folder: The path to the folder that will house the model to be training
      model_name: The name of the folder that will house all the data needed to
        classify with the trained model. This folder will sit directly within parent_folder.
        In other words, the full path to the folder with the saved model's data is
        'path_to_parent_folder/model_name'.
  
    Flags:
      [--isfile]: Pass this flag if the data field is a filepath and not a JSON formatted string.
      [--verbose_train]: Toggles verbose prints during training. Mostly relevant for keras neural networks.
      [--verbose_eval]: Toggles verbose evaluation prints.
      [--vv]: Toggles verbose_train and verbose_eval.
      [--display]: Displays all models available for training and then terminates immediatley,
        no matter what arguments have been passed.

    Examples:
      # standard
      train.py rnn data_as_a_json_formatted_string path/to/parent my_model
      # with data as a file
      train.py dnn path/to/my/data path/to/parent my_model --isfile
      # with verbose train and evaluation
      train.py rnn data_as_a_json_formatted_string path/to/parent my_model --vv

evaluate.py usage:

    Description: Loads the evaluation of a model saved during train. Currently does not allow the evaluation of performance on new data. This will output accuracy data for each class as well as an accuracy summary (average, weighted average by number of data points in each class, and median).

    evaluate.py path_to_parent_folder model_name [--readable]

    Mandatory Arguments (given in order):
      path_to_parent_folder: The path to the folder that will house the model to be training
      model_name: The name of the folder that will house all the data needed to
        classify with the trained model. This folder will sit directly within parent_folder.
        In other words, the full path to the folder with the saved model's data is
        path_to_parent_folder/model_name'.

    Flags:
      [--readable]: Toggles human readable output instead of outputting a JSON formatted string.

    Examples:
      # Machine readable (json output). Would be called by a script.
      evaluate.py path/to/parent my_model
      # Human readable (pretty prints)
      evaluate.py path/to/parent my_model --readable

classify.py usage:

    Description: This script allows you to load a previously trained model (trained with the train.py script) and classify new data. It's default behavior returns a single classification for all of the data passed. In this case, it classifies on each datapoint and then returns the most likely class(es) that all those datapoints would have come from.

    classify.py data path_to_parent_folder model_name [--isfile] [--verbose_train] [--verbose_eval] [--vv] [--display]

    Mandatory Arguments (given in order):
      model_type: The type of model to train (ie 'dnn', 'rnn', 'random_forest')
      data_to_train_on: The data to train on. Should be a JSON formatted string or
        a file housing a JSON formatted string.
      path_to_parent_folder: The path to the folder that will house the model to
        be training
      model_name: The name of the folder that will house all the data needed to
        classify with the trained model. This folder will sit directly within
        parent_folder. In other words, the full path to the folder with the saved
        model's data is 'path_to_parent_folder/model_name'.
  
    Optional Argument:
      group_size: The number of guesses to return (ordered most to least likely)
      window_size: The size of the window to use while classifying. This is used
        in the default behavior (only consider the last window_size points in
        the classification) and --trailing behavior (only consider the last
        window_size points in each datapoint's classification). NOTE: NOT YET IMPLEMENTED.

    Flags:
      [--isfile]: Pass this flag if the data field is a filepath and not a JSON
        formatted string.
      [--independent]: Returns a prediction on each datapoint. Does not accumulate
        predictions. NOTE: CURRENTLY NOT IMPLEMENTED.
      [--trailing]: Returns a prediction on each datapoint, but still uses information
        from previous data points (within a sliding window) to classify.
      [--readable]: Prints a human readable classification instead of a machine readable one.
      [--persistant]: Opens the file and persists as a REPL until it receives EOF or 'close'.
        NOTE: CURRENTLY NOT IMPLEMENTED.

    Examples:
      # standard
      classify.py data_as_a_json_formatted_string path/to/parent my_model
      # with data as a file
      classify.py path/to/my/data path/to/parent my_model --isfile
      # return a classification on each data point but still use previous classifications to inform
      classify.py data_as_a_json_formatted_string path/to/parent my_model --trailing
      # return a classification on each data point, ignore previous classifications
      classify.py data_as_a_json_formatted_string path/to/parent my_model --independent
      # return a classification consisting of the two most likely classes (in descending order)
      classify.py data_as_a_json_formatted_string path/to/parent my_model 2
      # return a classification informed only by the last 10 datapoints. Note that
      #   you need to pass a value to group_size in order to pass an argument to window_size.
      #   This is due to a peculiarity of the parsing module I am using.
      classify.py data_as_a_json_formatted_string path/to/parent my_model 1 10

## A Conceptual Overview

A data science pipeline generally follows these steps:

![pipeline](documentation/Photos/ML_Funnel.JPG)

DATA COLLECTION AND TRAINING PHASE:

1. Choose what data is relevant to your problem (not within the scope of this repo)
2. Begin to collect that data (not within the scope of this repo)
3. Parse it into a representation that is useful for data science (likely numpy or pandas. This repo expects a json formatted string with features and class labels.)
4. Preprocess the data
    * Feature engineering (Transform existing values within their predefined feature and/or combine existing values into new features. You end with the same features and possibly some additional ones).
    * Data normalization (Remove differences in magnitude and range. Example: Gaussian Normalization, which shifts the average of each feature to 0 and the standard devation of each feature to 1).
    * Feature selection (Choose the relevant features to use for training. Often called dimension reduction. You end this step with less features than you started with. Examples: PCA, SVD, Autoencoder).
5. Train the model
6. Evaluate the model.
    * Note that there is often a loop between train and evaluate where either a data scientist or an automated system trains the model, evaluates it, and then either chooses a different model (manually or by, for example, a genetic algorithm) or tunes the existing one (by changing hyperparameters). If you can get a consistent stream of labeled data it is recommended that you revisit this loop regularly, testing the performance of your model against new data and possibly retraining your model on the new dataset. Training on a larger dataset is always better (except for time constraints). Also, training on the newest data can be useful in certain tasks, as it can help your model adapt to a possibly changing environment. For example, if I am doing classification of rooms by wifi signals, my router signal strength may be dropping, or I may have put in a new wall or a new router. These changes will affect my ability to classify accurately, can only be ntocied through continuous evaluation, and only be corrected through continuous learning.
7. Save the trained model.

CLASSIFICATION PHASE:

1. Load the trained model.
2. Parse it into a representation that is useful for data science (likely numpy or pandas. This repo expects a json formatted string with features and class labels.)
3. Preprocess incoming data in the same manner that data was preprocessed during training. THIS IS VERY IMPORTANT. If you fail to do this, the model will be seeing data of a different form than that it was trained on which will adversely affect performance.
4. Classify the data.

This repo covers steps 3-7 of the training phase and all steps of the classification phase.

Each step is represented by either a single function or a single object, and can therefore be swapped out very easily. You can imagine that each step of the funnel is a lego block than can be easily swapped in and out.

The objects are implemented as such because they need to save state. I use two objects - a normalizer object (which must save normalization data), and an encoder object (which translates data from the representation of the outside world to that of the machine learning model and back again). These two objects are wrapped in a Preprocessor object, which is what mostly passed between functions and is what is saved at the end of the training phase.

The rest of the model is implemented in functions with (almost) no side effects. Even when there are side effects (which are necessary to write data to the normalizer, encoder, and trained model), I attempt to also return the modified object to maintain the semblance of a functional style.

## Creating Your Own Custom Model

### Developing Your Custom Model

In order to create a custom model, you should subclass Model (at sample/models/model). Model is an abstract class that contains almost all of the functionality necessary to train, save, and classify. Any subclass of model must implement:

* Training
  * select_mode
    * trains a model on data
  * save_model
    * saves a trained model
  * evaluate_in_pipeline
    * evaluates the trained model at the end of the training pipeline
* Classification
  * classify
    * classifies incoming data

Your custom model MUST also define a static variable MODEL_TYPE = 'my_model'. This should be the SAME VALUE YOU USE IN THE DICTIONARY WITHIN model_list.py (explained below the next code snippet). It is also recommended that a new model specifies a default encoder and a default normalizer in the constructor of the form (taken from the Dnn class at sample/models/dnn). Note that you can also overload any of the defaults in the Model class, and that overloading any given method is like overloading a certain slice of the funnel depicted above.

```python
'''
REQUIRED STATIC VARIABLE
'''
MODEL_TYPE = 'my_model'

'''
RECOMMENDED STATIC VARIABLE. If not supplied, the defaults defined in Model will be used
'''
transformation = staticmethod(lambda x : (x + 100) % 100)
preprocessor = Preprocessor(encoder=Categorical_encoder(), normalizer=Gaussian_normalizer(),
  transformation=transformation)

'''
REQUIRED, IN THIS FORM: Constructor
'''

def __init__(self, model=None, preprocessor=None):
    if preprocessor == None:
      preprocessor = self.preprocessor
    super(Dnn, self).__init__(model=model, preprocessor=preprocessor)
    self.accumulated_pred = np.zeros(preprocessor.get_encoder().get_num_labels())
```

The preprocessor (referenced at self.preprocessor) should contain instantiated versions of its encoder and normalizer. The if statement is necessary because python does not let you reference staticly defined variables when defining a default value of a constructor, and so we must use the 'set default to None and then check with an if statement' hack. The call to super should look like

```python
super(My_model_type, self).__init__(model=model, preprocessor=preprocessor)
```

and handles some finicky python things with regards to passing methods housed within classes. You should not need to touch the Model constructor, and if you pass it a valid model and preprocessor (as shown above), everything should work out swimmingly.

Lastly, you must import your new model to model_list.py (at sample/models/model_list.py). This is what train imports to map the model_type (aka string) you pass during training to an actual class. Once you import it, you must add it to the dictionary within, with the key being the string you want to pass as model_type during training and the value being the class of your model. For example, if you want to call your model during train by the type 'my_model':

```python
# previously implemented models
from .dnn import Dnn
from .rnn import Rnn
# my new model
from .my_model import My_model
# the dictionary used by train.py
models = {
  'dnn' : Dnn,
  'rnn' : Rnn
  'my_model' : My_model
}
```

### Testing Your Custom Model

If you follow the method contracts laid out in Model, the repo should have some prebuilt tests that can verify some basic features of your model. You can run them by running 'pytest' from terminal from the **top directory of the repo** (weird things may happen otherwise).

These tests will run your whole model pipeline (it will run training_pipeline, load the saved model, and then attempt to classify some data). These tests will also assert some basic features about your data (train_test_split behaves reasonably well, the outputs of your submethods make some sort of sense). It is HIGHLY RECOMMENDED that you implement some of your own tests, as these are necessary (but not sufficient) to guarantee functionality. Also note that much about machine learning is not automatically testable in the traditional sense (it's hard to judge performance), so be sure to pay attention the the evaluation report printed at the end of training your model. You can reload the report using evaluation.py.

The tests can take a while because of the training of the neural networks, so if you want to do them more quickly, you should make them skip Dnn and Rnn.