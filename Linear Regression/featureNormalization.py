import numpy as np

def featureNormalization(X):
    shape = X.shape
    mu = np.empty(shape[0] - 1)
    sigma = np.empty(shape[0] - 1)
    for idx in range(1, shape[0]):
        mu[idx - 1] = np.mean(X[idx][:])
        sigma[idx - 1] = np.std(X[idx][:])
        X[idx][:] = (X[idx][:] - mu[idx - 1]) / sigma[idx - 1]
    return (X, mu, sigma)


#
# if __name__ == "__main__":
#     x = [1, 2]
#     print featureNormalization(x)
#     print x