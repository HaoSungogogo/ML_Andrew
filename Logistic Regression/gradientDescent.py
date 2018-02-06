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

