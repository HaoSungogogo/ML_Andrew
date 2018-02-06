import numpy as np
import readdata
import plotdata
from mapFeature import mapFeature
from computeCost import computeRegularizedCost
import gradientDescent
from plotdata import plotReg

if __name__ == "__main__":
    (x, y, num) = readdata.readData2()
    plotdata.plotPoints(x, y)
    degree = 6
    (X, numOfFeatures) = mapFeature(x, degree)
    theta = np.zeros(numOfFeatures + 1)
    lam = 1
    print computeRegularizedCost(X, y, theta, lam)
    iterations = 100000
    alpha = 0.001
    gradientDescent.gradientDescent(X, y, theta, alpha, iterations)
    plotReg(x, y, theta, degree)