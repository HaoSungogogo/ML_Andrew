# regression mode: define the relationship between input and output, and use a new input combined with such relation to
# predict corresponding output

# training set -> learning algorithm -> prediction

import numpy as np
import matplotlib.pyplot as plot
from computeCost import computeCost
from gradientDescent import gradientDescent
from featureNormalization import featureNormalization
import readData

numOfFeatures = 2

if __name__ == "__main__":
    iterations = 0
    alpha = 0.01
    if numOfFeatures == 1:
        x, y, numExamples = readData.readSingleFeature()
        X = np.ones((numOfFeatures + 1, numExamples))
        X[1:][:] = x[:][:]
        theta = np.zeros(numOfFeatures + 1)
        print computeCost(X, y, theta)
        iterations = 1500
        theta = gradientDescent(X, y, theta, alpha, iterations)
        plot.plot(x, y, 'o', x, np.dot(theta, X))
        plot.show()
    elif numOfFeatures == 2:
        x, y, numExamples = readData.readMultiFeature()
        X = np.ones((numOfFeatures + 1, numExamples))
        X[1:][:] = x[:][:]
        theta = np.zeros(numOfFeatures + 1)
        (X_norm, mu, sigma) = featureNormalization(X)
        print computeCost(X_norm, y, theta)
        iterations = 500
        theta = gradientDescent(X, y, theta, alpha, iterations)