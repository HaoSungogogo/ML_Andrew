import numpy as np
import csv

def readData():
    x = np.empty([2, 1])
    temp = np.empty([2, 1])
    y = np.empty(1, dtype = float)
    with open('ex2data1.txt', 'r') as file:
        datafile = csv.reader(file)
        count = 0
        for line in datafile:
            if count == 0:
                x[0][0] = line[0]
                x[1][0] = line[1]
                y[0] = np.float_(line[2])
            else:
                temp[0][0] = line[0]
                temp[1][0] = line[1]
                x = np.append(x, temp, axis=1)
                y = np.append(y, np.float_(line[2]))
            count += 1
    return (x, y, count)

def readData2():
    x = np.empty([2, 1])
    temp = np.empty([2, 1])
    y = np.empty(1, dtype=float)
    with open('ex2data2.txt', 'r') as file:
        datafile = csv.reader(file)
        count = 0
        for line in datafile:
            if count == 0:
                x[0][0] = line[0]
                x[1][0] = line[1]
                y[0] = np.float_(line[2])
            else:
                temp[0][0] = line[0]
                temp[1][0] = line[1]
                x = np.append(x, temp, axis=1)
                y = np.append(y, np.float_(line[2]))
            count += 1
    return (x, y, count)





        # if __name__ == "__main__":
#     (x, y, count) = readData()
#     print x