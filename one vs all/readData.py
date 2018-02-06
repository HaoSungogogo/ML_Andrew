import scipy.io as io

def readData():
    mat = io.loadmat("ex3data1.mat")
    y = mat['y']
    X = mat['X']
    return (X, y)
