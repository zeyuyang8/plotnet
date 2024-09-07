''' Source: https://github.com/tensorflow/playground/blob/master/src/dataset.ts '''


import random
import math
from scipy.interpolate import interp1d
import numpy as np


NUM_SAMPLES_CLASSIFY = 500
NUM_SAMPLES_REGRESS = 1200
TYPE_OF_FEATURES = {
    "x": {"f": lambda x: x[0], "label": "X_1"},
    "y": {"f": lambda x: x[1], "label": "X_2"},
    "x squared": {"f": lambda x: x[0] * x[0], "label": "X_1^2"},
    "y squared": {"f": lambda x: x[1] * x[1], "label": "X_2^2"},
    "x times y": {"f": lambda x: x[0] * x[1], "label": "X_1X_2"},
    "sin x": {"f": lambda x: math.sin(x[0]), "label": "sin(X_1)"},
    "sin y": {"f": lambda x: math.sin(x[1]), "label": "sin(X_2)"}
}


def rand_uniform(left, right):
    '''
    Returns a sample from a uniform [left, right] distribution.
    '''
    return random.random() * (right - left) + left


def rand_normal(mean, variance):
    '''
    Returns a sample from a normal distribution.
    '''
    while True:
        v1 = 2 * random.random() - 1
        v2 = 2 * random.random() - 1
        s = v1 * v1 + v2 * v2
        if s <= 1:
            break
    result = math.sqrt(-2 * math.log(s) / s) * v1
    return mean + math.sqrt(variance) * result


def euclidean_distance(a, b):
    '''
    Returns the euclidean distance between two points in space.
    '''
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def shuffle(array):
    '''
    Shuffles the array using Fisher-Yates algorithm.
    '''
    counter = len(array)
    temp = 0
    index = 0

    # While there are elements in the array
    while counter > 0:
        # Pick a random index
        index = math.floor(random.random() * counter)
        # Decrease counter by 1
        counter -= 1
        # And swap the last element with it
        temp = array[counter]
        array[counter] = array[index]
        array[index] = temp


def regression_plane_generator(num_samples, noise=0):
    '''
    Generates a random regression plane data set.
    '''
    radius = 6
    label_scale = interp1d([-10, 10], [-1, 1], "linear",
                           bounds_error=False, fill_value="extrapolate")

    def get_label(x, y):
        return label_scale(x + y)

    points = []
    for dummy in range(num_samples):
        x = num_samples(-radius, radius)
        y = num_samples(-radius, radius)
        noiseX = rand_uniform(-radius, radius) * noise
        noiseY = rand_uniform(-radius, radius) * noise
        label = get_label(x + noiseX, y + noiseY)
        points.append([x, y, label])

    return points


def regression_gaussian_generator(num_samples, noise=0):
    '''
    Generates a random regression Gaussian data set.
    '''
    points = []
    label_scale = interp1d([0, 2], [1, 0], "linear",
                           bounds_error=False, fill_value=(1, 0))
    gaussians = [[-4, 2.5, 1],
                 [0, 2.5, -1],
                 [4, 2.5, 1],
                 [-4, -2.5, -1],
                 [0, -2.5, 1],
                 [4, -2.5, -1]]

    def get_label(x, y):
        ''' Choose the one that is maximum in abs value. '''
        label = 0
        for cx, cy, sign in gaussians:
            newLabel = sign * label_scale(euclidean_distance([x, y], [cx, cy]))
            if abs(newLabel) > abs(label):
                label = newLabel
        return label

    radius = 6
    for dummy in range(num_samples):
        x = rand_uniform(-radius, radius)
        y = rand_uniform(-radius, radius)
        noise_x = rand_uniform(-radius, radius) * noise
        noise_y = rand_uniform(-radius, radius) * noise
        label = get_label(x + noise_x, y + noise_y)
        points.append([x, y, label])
    return points


def classification_gaussian_generator(num_samples, noise=0):
    '''
    Generates a random classification Gaussian data set.
    '''
    variance_scale = interp1d([0, .5], [0.5, 4], "linear")
    variance = variance_scale(noise)
    points = []

    def get_gauss(cx, cy, label):
        for dummy in range(num_samples // 2):
            x = rand_normal(cx, variance)
            y = rand_normal(cy, variance)
            points.append([x, y, label])

    get_gauss(2, 2, 1)
    get_gauss(-2, -2, 1)
    return points


def classification_spiral_generator(num_samples, noise=0):
    '''
    Generates a random classification spiral data set.
    '''
    points = []
    n = num_samples // 2

    def get_spiral(delta_t, label):
        for i in range(n):
            r = i / n * 5
            t = 1.75 * i / n * 2 * math.pi + delta_t
            x = r * math.sin(t) + rand_uniform(-1, 1) * noise
            y = r * math.cos(t) + rand_uniform(-1, 1) * noise
            points.append([x, y, label])

    get_spiral(0, 1)  # Positive examples
    get_spiral(math.pi, -1)  # Negative examples
    return points


def classification_circle_generator(num_samples, noise=0):
    '''
    Generates a random classification circle data set.
    '''
    points = []
    radius = 5
    n = num_samples // 2

    def get_circle_label(p, center):
        if euclidean_distance(p, center) < (radius * 0.5):
            return 1
        return -1

    def get_circle(radius_range):
        left, right = radius_range
        for dummy in range(n):
            r = rand_uniform(left, right)
            angle = rand_uniform(0, 2 * math.pi)
            x = r * math.sin(angle)
            y = r * math.cos(angle)
            noise_x = rand_uniform(-radius, radius) * noise
            noise_y = rand_uniform(-radius, radius) * noise
            label = get_circle_label([x + noise_x, y + noise_y], [0, 0])
            points.append([x, y, label])

    get_circle([0, radius * 0.5])   # Positive points inside the circle
    get_circle([radius * 0.7, radius])   # Negative points outside the circle
    return points


def classification_xor_generator(num_samples, noise=0):
    '''
    Generates a random classification XOR data set.
    '''
    def get_xor_label(p):
        if p[0] * p[1] > 0:
            return 1
        return -1

    points = []
    for dummy in range(num_samples):
        padding = 0.3
        x = rand_uniform(-5, 5)
        if x > 0:
            x += padding
        elif x <= 0:
            x += -padding

        y = rand_uniform(-5, 5)
        if y > 0:
            y += padding
        elif y <= 0:
            y += -padding

        noise_x = rand_uniform(-5, 5) * noise
        noise_y = rand_uniform(-5, 5) * noise
        label = get_xor_label([x + noise_x, y + noise_y])
        points.append([x, y, label])

    return points


MAPPING = {
    "reg_plane": regression_plane_generator,
    "reg_gauss": regression_gaussian_generator,
    "class_gauss": classification_gaussian_generator,
    "spiral": classification_spiral_generator,
    "circle": classification_circle_generator,
    "xor": classification_xor_generator
}


def generate_data(num_samples, choice, features, noise, perc_train, seed=2023):
    '''
    Generates a data set.
    Args:
        num_samples - int of number of samples to generate.
        choice - choice of function that generates the data.
        features - list of strings representing features to use.
        noise - int of noise level.
        perc_train - float of percentage of data to use for training.
        seed - int of seed for random number generator.
    Returns a tuple of two tuples, where the first tuple is the training data,
    and the second tuple is the testing data. Each tuple contains two numpy arrays.
        train_feature - numpy array of training features.
        train_label - numpy array of training labels.
        test_feature - numpy array of testing features.
        test_label - numpy array of testing labels.
    '''
    if seed:
        random.seed(seed)

    generator = MAPPING[choice]
    data = generator(num_samples, noise)
    shuffle(data)

    split_idx = math.floor(len(data) * perc_train)
    train_data = np.array(data[:split_idx])
    test_data = np.array(data[split_idx:])
    train_feature, train_label = train_data[:, [0, 1]], train_data[:, 2]
    test_feature, test_label = test_data[:, [0, 1]], test_data[:, 2]
    train_transformed_feature = []
    test_transformed_feature = []

    for feature in features:
        func = TYPE_OF_FEATURES[feature]["f"]
        temp1 = np.apply_along_axis(func, 1, train_feature)
        temp2 = np.apply_along_axis(func, 1, test_feature)
        train_transformed_feature.append(temp1)
        test_transformed_feature.append(temp2)
    train_transformed_feature = np.vstack(train_transformed_feature).T
    test_transformed_feature = np.vstack(test_transformed_feature).T

    return (train_transformed_feature, train_label), (test_transformed_feature, test_label)
