import numpy as np
from sigmoid import sigmoid

def computeRegularizedCost(theta,X,y,lam):
    m=len(y)
    z = np.dot(X,theta)
    h = sigmoid(z)
    J = ( (lam*sum(theta[1:]*theta[1:])/2.0) +
          sum((-y*np.log(h))-((1-y)*np.log(1-h))) )/m
    return J

def regularizedLogisticDerivative(theta, X, y, lam):
    m = len(y)
    numOfFeatures = len(theta)
    derivative = np.empty(numOfFeatures)
    z = np.dot(X, theta)
    h = sigmoid(z)
    derivative[0] = sum((h - y) * X[:, 0]) / m
    for i in range(1, numOfFeatures):
        derivative[i] = (sum((h - y) * X[:, i]) + lam * theta[i]) / m
    return derivative

