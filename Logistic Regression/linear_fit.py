# 0/1 classification problem is hard for us if we adopt linear regression, so we use logistic regression to
# accomplish this, with use of sigmoid function

import numpy as np
import readdata
import plotdata
from computeCost import computeCost
from gradientDescent import gradientDescent


if __name__ == "__main__":
    (x, y, num) = readdata.readData()
    plotdata.plotPoints(x, y)
    shape = x.shape
    numOfFeatures = shape[0]
    X = np.ones([numOfFeatures + 1, num])
    X[1:,:] = x[:,:]
    theta = np.zeros(numOfFeatures + 1)
    print computeCost(X, y, theta)
    iterations = 100000
    alpha = 0.001
    plotdata.plotTheta(x, y, gradientDescent(X, y, theta, alpha, iterations))