import numpy as np
import costFunction
from scipy.optimize import fmin_bfgs

def oneVsAll(X, y, numOfLables, lam):
    dim = X.shape
    row = dim[0]
    col = dim[1]

    all_theta = np.zeros([numOfLables, col + 1])
    initial_theta = np.zeros(col + 1)

    updatedX = np.ones([row, col + 1])
    updatedX[:, 1:] = X[:,:]
    y = y[:, 0]

    for i in range(numOfLables):
        newY = np.zeros(y.size)
        if i == 0:
            idx = np.where(y == 10)
        else:
            idx = np.where(y == i)
        newY[idx] = 1
        theta = fmin_bfgs(costFunction.computeRegularizedCost, initial_theta,
                          fprime=costFunction.regularizedLogisticDerivative, args=(updatedX, newY, lam))
        all_theta[i, :] = theta[:]
    return all_theta