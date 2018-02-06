import numpy as np
from computeCost import computeCost
from sigmoid import sigmoid
import matplotlib.pyplot as plt

def gradientDescent(X, y, theta, alpha, numOfIterations):
    flag = 1
    num = len(y)
    numOfFeatures = len(theta)
    history = np.zeros(numOfIterations)
    for i in range(numOfIterations):
        z = np.dot(theta, X)
        h = sigmoid(z)
        for idx in range(numOfFeatures):
            theta[idx] -= alpha * sum((h - y) * X[idx][:]) / num
        if flag:
            history[i] = computeCost(X, y, theta)
    if flag:
        plt.plot(history)
        plt.show()
    return theta

# def logisticDerivative(X, y, theta):
#     num = len(y)
#     numOfFeatures = len(theta)
#     z = np.dot(theta, X)
#     h = sigmoid(z)
#     derivative = np.empty(numOfFeatures)
#     for idx in range(numOfFeatures):
#         derivative[idx] = sum((h - y) * X[idx][:]) / num
#     return derivative

def logisticDerivative(theta,X,y):
    m = len(y)
    features = len(theta)
    derive = np.empty(features)
    z = np.dot(theta,X)
    h = sigmoid(z)
    for jj in range(features):
        derive[jj] = sum((h-y)*X[jj,:])/m
    return derive
