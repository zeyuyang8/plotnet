import random
import math
from scipy.interpolate import interp1d
import numpy as np

# Helper functions
def randUniform(a, b):
    ''' 
    Returns a sample from a uniform [a, b] distribution. 
    Uses the seedrandom library as the random generator.
    '''
    return random.random() * (b - a) + a

def normalRandom(mean, variance):
    '''
    Samples from a normal distribution. Uses the seedrandom library as the 
    random generator.
    '''

    while True:
        v1 = 2 * random.random() - 1
        v2 = 2 * random.random() - 1
        s = v1 * v1 + v2 * v2
        if s <= 1:
            break
    
    result = math.sqrt(-2 * math.log(s) / s) * v1
    return mean + math.sqrt(variance) * result

def dist(a, b):
    ''' Returns the eucledian distance between two points in space. '''

    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)

def shuffle(array):
    ''' 
    Shuffles the array using Fisher-Yates algorithm. 
    Uses the seedrandom library as the random generator. 
    '''
    if type(array) == tuple:
        return -1
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

# Data set generators

def regressPlane(numSamples, noise=0):
    radius = 6
    labelScale = interp1d([-10, 10], [-1, 1], "linear", bounds_error=False, fill_value="extrapolate")
    getLabel = lambda x, y: labelScale(x + y)
    points = []

    for dummy in range(numSamples):
        x = randUniform(-radius, radius)
        y = randUniform(-radius, radius)
        noiseX = randUniform(-radius, radius) * noise
        noiseY = randUniform(-radius, radius) * noise
        label = getLabel(x + noiseX, y + noiseY)
        points.append([x, y, label])

    return points

def regressGaussian(numSamples, noise=0):
    points = []
    labelScale = interp1d([0, 2], [1, 0], "linear", bounds_error=False, fill_value=(1, 0))

    gaussians = [
        [-4, 2.5, 1],
        [0, 2.5, -1],
        [4, 2.5, 1],
        [-4, -2.5, -1],
        [0, -2.5, 1],
        [4, -2.5, -1]
        ]
    
    def getLabel(x, y):
        ''' Choose the one that is maximum in abs value. '''

        label = 0
        for cx, cy, sign in gaussians:
            newLabel = sign * labelScale(dist([x, y], [cx, cy]))
            if abs(newLabel) > abs(label):
                label = newLabel
        return label

    radius = 6
    
    for dummy in range(numSamples):
        x = randUniform(-radius, radius)
        y = randUniform(-radius, radius)
        noiseX = randUniform(-radius, radius) * noise
        noiseY = randUniform(-radius, radius) * noise
        label = getLabel(x + noiseX, y + noiseY)
        points.append([x, y, label])
    
    return points


def classifyTwoGaussData(numSamples, noise=0):
    variance_scale = interp1d([0, .5], [0.5, 4], "linear")
    variance = variance_scale(noise)
    points = []

    def genGauss(cx, cy, label):
        for dummy in range(numSamples // 2):
            x = normalRandom(cx, variance)
            y = normalRandom(cy, variance)
            points.append([x, y, label])

    genGauss(2, 2, 1)
    genGauss(-2, -2, -1)

    return points


def classifySpiralData(numSamples, noise=0):
    points = []
    n = numSamples // 2

    def genSpiral(deltaT, label):
        for i in range(n):
            r = i / n * 5
            t = 1.75 * i / n * 2 * math.pi + deltaT
            x = r * math.sin(t) + randUniform(-1, 1) * noise
            y = r * math.cos(t) + randUniform(-1, 1) * noise
            points.append([x, y, label])

    genSpiral(0, 1) # Positive examples
    genSpiral(math.pi, -1) # Negative examples

    return points

def classifyCircleData(numSamples, noise=0):
    points = []
    radius = 5
    def getCircleLabel(p, center):
        if dist(p, center) < (radius * 0.5):
            return 1
        return -1

    # Generate positive points inside the circle.
    for dummy in range(numSamples // 2):
        r = randUniform(0, radius * 0.5)
        angle = randUniform(0, 2 * math.pi)
        x = r * math.sin(angle)
        y = r * math.cos(angle)
        noiseX = randUniform(-radius, radius) * noise
        noiseY = randUniform(-radius, radius) * noise
        label = getCircleLabel([x + noiseX, y + noiseY], [0, 0])
        points.append([x, y, label])
    
    # Generate negative points inside the circle.
    for dummy in range(numSamples // 2):
        r = randUniform(radius * 0.7, radius)
        angle = randUniform(0, 2 * math.pi)
        x = r * math.sin(angle)
        y = r * math.cos(angle)
        noiseX = randUniform(-radius, radius) * noise
        noiseY = randUniform(-radius, radius) * noise
        label = getCircleLabel([x + noiseX, y + noiseY], [0, 0])
        points.append([x, y, label])
    return points

def classifyXORData(numSamples, noise):
    def getXORLabel(p):
        if p[0] * p[1] >= 0:
            return 1
        return -1
    
    points = []
    for dummy in range(numSamples):
        padding = 0.3
        x = randUniform(-5, 5)
        if x > 0:
            x += padding
        elif x <= 0:
            x += -padding

        y = randUniform(-5, 5)
        if y > 0:
            y += padding
        elif y <= 0:
            y += -padding

        noiseX = randUniform(-5, 5) * noise
        noiseY = randUniform(-5, 5) * noise
        label = getXORLabel([x + noiseX, y + noiseY])
        points.append([x, y, label])
        
    return points

def classifyTwoPoints(dummy_numSamples, dummy_noise):
    return ([-2, -2, -1], [2, 2, 1], [-2, -2, -1], [2, 2, 1])