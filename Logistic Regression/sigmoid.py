import numpy as np

def sigmoid(z):
    return 1 / (1.0 + np.exp(-z))