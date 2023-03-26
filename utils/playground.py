import math
import random
import numpy as np
from dataset import *

# Global variables
NUM_SAMPLES_CLASSIFY = 500
NUM_SAMPLES_REGRESS = 1200
INPUT = {
    "x": {"f": lambda x: x[0], "label": "X_1"},
    "y": {"f": lambda x: x[1], "label": "X_2"},
    "xSquared": {"f": lambda x: x[0] * x[0], "label": "X_1^2"},
    "ySquared": {"f": lambda x: x[1] * x[1], "label": "X_2^2"},
    "xTimesY": {"f": lambda x: x[0] * x[1], "label": "X_1X_2"},
    "sinX": {"f": lambda x: math.sin(x[0]), "label": "sin(X_1)"},
    "sinY": {"f": lambda x: math.sin(x[1]), "label": "sin(X_2)"}
}    

def generateData(state, seed=0.29963):
    '''
    state is a dictionary.
    '''
    if seed:
        random.seed(seed)
    
    if state["problem"] == "REGRESSION":
        numSamples = NUM_SAMPLES_REGRESS
        generator = state["regDataset"]
    elif state["problem"] == "CLASSIFICATION":
        numSamples = NUM_SAMPLES_CLASSIFY
        generator = state["dataset"]

    data = generator(numSamples, state["noise"] / 100)

    # Shuffle data
    shuffle(data)

    # Split into train and test data.
    splitIndex = math.floor(len(data) * state["percTrainData"] / 100);

    # Train data
    train_data = np.array(data[:splitIndex])
    train_feature, train_label = train_data[:, [0, 1]], train_data[:, 2]

    train_transformed_feature = []
    for feature in state["inputFeature"]:
        func = INPUT[feature]["f"]
        temp = np.apply_along_axis(func, 1, train_feature)
        train_transformed_feature.append(temp)
    
    train_transformed_feature = np.vstack(train_transformed_feature).T

    # Test data
    test_data = np.array(data[splitIndex:])
    test_feature, test_label = test_data[:, [0, 1]], test_data[:, 2]

    test_transformed_feature = []
    for feature in state["inputFeature"]:
        func = INPUT[feature]["f"]
        temp = np.apply_along_axis(func, 1, test_feature)
        test_transformed_feature.append(temp)
    
    test_transformed_feature = np.vstack(test_transformed_feature).T

    return (train_transformed_feature, train_label), (test_transformed_feature, test_label)

