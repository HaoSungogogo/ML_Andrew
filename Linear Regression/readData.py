import numpy as np
import csv
import matplotlib.pyplot as plot

def readSingleFeature():
    num = 97
    x = np.empty([num], dtype = float)
    y = np.empty([num], dtype= float)
    with open('ex1data1.txt', 'r') as data_one:
        data_file = csv.reader(data_one)
        idx = 0
        for line in data_file:
            x[idx] = line[0]
            y[idx] = line[1]
            idx += 1
    # plot.plot(x, y, 'o')
    return (x, y, num)

def readMultiFeature():
    num = 47
    x = np.empty([2, num], dtype = float)
    y = np.empty([num], dtype = float)
    with open('ex1data2.txt', 'r') as data_two:
        data_file = csv.reader(data_two)
        idx = 0
        for line in data_file:
            x[0][idx] = line[0]
            x[1][idx] = line[1]
            y[idx] = line[2]
            idx += 1
    return (x, y, num)




# if __name__ == "__main__":
#     x = np.empty([2, 3], dtype=float)
#     print x