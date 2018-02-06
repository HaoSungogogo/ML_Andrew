import numpy as np
from sigmoid import sigmoid

def computeCost(X, y, theta):
    num = len(y)
    z = np.dot(theta, X)
    h = sigmoid(z)
    J = sum(-y * np.log(h) -((1 - y) * np.log(1 - h))) / num
    return J


def computeRegularizedCost(X, y, theta, lam):
    num = len(y)
    z = np.dot(theta, X)
    h = sigmoid(z)
    J = (sum(-y * np.log(h) - ((1 - y) * np.log(1 - h))) + lam * sum(theta[1:] * theta[1:]) / 2.0) / num
    return J