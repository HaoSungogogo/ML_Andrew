import readData
from oneVsAll import oneVsAll
import pickle
from predictOneVsAll import predictOneVsAll

firstRun = 0

if __name__ == "__main__":
    (X, y) = readData.readData()
    numOfLabels = 10
    lam = 0.1
    if firstRun:
        all_theta = oneVsAll(X, y, numOfLabels, lam)
        with open('thetas.pickle', 'w') as f:
            pickle.dump([all_theta], f)
    else:
        with open('thetas.pickle') as f:
            [all_theta] = pickle.load(f)

    prediction = predictOneVsAll(X, y, all_theta, numOfLabels)
    print "One vs All determines the handwriting correctly on the training set " + str(
        100 * prediction) + "% of the time."