import numpy as np



def computeCost(X, y, theta):
    m = len(y)
    J = sum((np.dot(theta, X) - y) * (np.dot(theta, X) - y)) / (2 * m)
    return J