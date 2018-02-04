import numpy as np
import matplotlib.pyplot as plot
from computeCost import computeCost

def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y)
    numOfFeatures = len(theta)
    debug = 0
    historyJ = np.zeros(iterations)
    for i in range(iterations):
        for idx in range(numOfFeatures):
            theta[idx] -= alpha * sum((np.dot(theta, X) - y) * X[idx][:]) / m
            if debug:
                historyJ[idx] = computeCost(X, y, theta)
    if debug:
        plot.plot(historyJ)
        plot.show()

    return theta