import numpy as np

def mapFeature(x, degree):
    num = x.shape[1]
    numOfFeatures = 0
    for i in range(1, degree + 1):
        numOfFeatures += i + 1
    output = np.ones([numOfFeatures + 1, num])
    idx = 1
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            output[idx][:] = (x[0][:]**(i - j)) * (x[1][:]**(j))
            idx += 1
    return (output, numOfFeatures)


