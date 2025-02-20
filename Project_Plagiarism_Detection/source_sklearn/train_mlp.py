from __future__ import print_function

import argparse
import os
import pandas as pd

# sklearn.externals.joblib is deprecated in 0.21 and will be removed in 0.23.
# from sklearn.externals import joblib
# Import joblib package directly
import joblib

# TODO: Import any additional libraries you need to define a model
from sklearn import tree
from sklearn.neural_network import MLPClassifier

# Provided model load function


def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")

    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")

    return model


# TODO: Complete the main code
if __name__ == '__main__':

    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job

    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str,
                        default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str,
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str,
                        default=os.environ['SM_CHANNEL_TRAIN'])

    # TODO: Add any additional arguments that you will need to pass into your model
    parser.add_argument('--hidden_layer_sizes', type=int, default=2)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--solver', type=str, default='adam')
    parser.add_argument('--random_state', type=int, default=1)
    parser.add_argument('--max_iter', type=int, default=300)

    # args holds all passed-in arguments
    args = parser.parse_args()

    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(
        training_dir, "train.csv"), header=None, names=None)

    # Labels are in the first column
    train_y = train_data.iloc[:, 0]
    train_x = train_data.iloc[:, 1:]

    ## --- Your code here --- ##
    hidden_layer_sizes = args.hidden_layer_sizes
    activation = args.activation
    solver = args.solver
    random_state = args.random_state
    max_iter = args.max_iter

    # TODO: Define a model
    # nn MLP Classifier
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                          activation=activation,
                          solver=solver,
                          random_state=random_state,
                          max_iter=max_iter)

    # TODO: Train the model
    model = model.fit(train_x, train_y)

    ## --- End of your code  --- ##

    # Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
