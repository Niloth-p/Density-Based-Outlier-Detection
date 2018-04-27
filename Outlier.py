# pylint: disable=invalid-name
import pickle
from math import inf
import pandas as pd
import numpy as np
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#import matplotlib.pyplot as plt

first_time = False
#parameters = [(50, 300) - 0.54, 0.6, (40, 600) - 0.438, 0.3, (75, 300), (100, 200), (200, 200), (200, 100), (400, 100)]
#parameters = [(10, 100), (20, 100), (30, 100), (40, 100), (50, 100)] - 0.38, 0.1
#parameters = [(10, 100) - 0.388, 0.094, (20, 200), (50, 200) -  - 0.414, 0.19, (50, 400), (100, 400) - 0.456, 0.3857]
parameters = [(60, 700), (40, 600), (50, 500), (100, 400), (50, 300)]
filename = 'german.data-numeric.txt'


def readData():
    '''
    Reads data from text file and stores as data frame
    '''
    df = pd.read_table(filename, header=None, delim_whitespace=True)
    df = df.iloc[:, :-1]
    df = (df - df.min()) / (df.max() - df.min())
    Y = df.iloc[:, -1]
    return (df, Y)


def mahanalobisdist(a, b):
    '''
    Calculates the mahalanobis distance
    between 2 points of the data
    '''
    temp = np.array([a, b]).T
    cov = np.cov(temp)
    delta = a - b
    inv = np.linalg.pinv(cov)
    mdist = np.dot(np.dot(np.transpose(delta), inv), delta)
    mdist = np.sqrt(mdist)
    return mdist


def createDistanceMatrix(data, first_timeval, N):
    '''
    Computes the distance matrix and
    writes to to a pickle file to save time
    on future runs
    '''
    distancematrix = np.zeros((N, N))
    if first_timeval:
        i = 0
        for value1 in data:
            j = 0
            for value2 in data:
                distancematrix[i][j] = mahanalobisdist(value1, value2)
                #print(distancematrix[i][j])
            j += 1
        i += 1

        f = open('distancematrix', 'wb')
        pickle.dump(distancematrix, f)
        f.close()

    else:
        f2 = open('distancematrix', 'rb')
        distancematrix = pickle.load(f2)
        f2.close()
    return distancematrix


def getLRD(N, distancematrix, k, data):
    '''
    Finds
    1. The KNN and hence the kdistance for each point
    i.e the distance to its kthNN,
    2. The number of points that fall within the k-distance neighbourhood
    3. Reachability distances
    4. lrd (local reachability density)
    for each point
    '''
    kdist = np.zeros(N)
    kneighbours = {}
    Numneighbours = 0
    lrd = np.zeros(N)

    for i in range(N):
        distancefrompoint = distancematrix[i]
        knn = np.partition(distancefrompoint, k-1)
        kdist[i] = knn[k-1]
        sort_index = np.argsort(distancefrompoint)

        j = 0
        temp = []
        for dist in distancefrompoint:
            if dist <= kdist[i]:
                temp.append(sort_index[j])
                Numneighbours += 1
            j += 1
        kneighbours[i] = temp

    reachabilitydistance = getReachabilityDistances(N, data, kdist, distancematrix)

    for i in range(N):
        sumOfReachabilityDistances = 0
        for value in kneighbours[i]:
            sumOfReachabilityDistances += reachabilitydistance[int(value)][i]
        if sumOfReachabilityDistances == 0:
            lrd[i] = inf
        lrd[i] = len(kneighbours[i])/sumOfReachabilityDistances

    return lrd


def getReachabilityDistances(N, data, kdist, distancematrix):
    '''
    Calculates the reachability distance
    between all pairs of points
    '''
    reachabilitydistance = np.zeros((N, N))
    i = 0
    for _ in data:
        j = 0
        for _ in data:
            reachabilitydistance[i][j] = max(kdist[i], distancematrix[i][j])
            j += 1
        i += 1
    return reachabilitydistance


def getAccuracy(outliers, Y, N, PrecisionList, RecallList):
    '''
    Gets the performace measures of the outlier detection done,
    in terms of Accuracy, Precision, Recall, F1-Score
    using true and false +ves and -ves
    '''
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    #testY = []
    for i, row in Y.iteritems():
        if i in outliers:
            #testY.append(1)
            if row == 1:
                tp += 1
            else:
                fp += 1
        else:
            #testY.append(0)
            if row == 1:
                fn += 1
            else:
                tn += 1
    print("True +ve:" + str(tp) + " True -ve:" + str(tn))
    print(" False +ve:" + str(fp) + " False -ve:" + str(fn))
    A = (tp + tn)/(tp + tn + fp + fn)
    P = (float(tp)/(tp + fp))
    R = (float(tp)/(tp + fn))
    f1 = 2*P*R/float(P+R)
    print("Accuracy : " + str(A) + " Precision : " + str(P) + " Recall : " + str(R) + " F1-Score : " + str(f1))
    print()
    PrecisionList.append(P)
    RecallList.append(R)
    #return testY


# def dimRedPlot(df, testY):
#     '''
#     Reduce dimensions to 2, then plot the points
#     of the obtained results, with outliers (i.e testY = 1)
#     highlighted in red and normal pts in blue
#     '''
#     lda = LDA(n_components=2)
#     lda_transformed = pd.DataFrame(lda.fit_transform(df, testY))

    # Plot normal points in blue and outliers in red
    # plt.scatter(lda_transformed[:][testY == 1], lda_transformed[:][testY == 1], label='Outliers', c='red')
    # plt.scatter(lda_transformed[testY == 0][0], lda_transformed[testY == 0][1], label='Normal points', c='blue')

    # #plt.legend(loc=3)
    # plt.show()


def main():
    '''
    Calls the functions to get distance matrix,
    the LRD, and the 1st O points after sorting of LRD
    and gets the Precision and Recall values
    '''
    df, Y = readData()
    i = 1
    PrecisionList = []
    RecallList = []
    data = df.values
    N = len(data)
    distancematrix = createDistanceMatrix(data, first_time, N)
    #O is the #of outliers
    for (k, O) in parameters:
        print("Experiment:", i, ", k =", k, ", num_outliers =", O)
        lrd = getLRD(N, distancematrix, k, data)
        sorted_outlier_factor_indexes = np.argsort(lrd)
        outliers = sorted_outlier_factor_indexes[-O:]
        getAccuracy(outliers, Y, N, PrecisionList, RecallList)
        i += 1
        #dimRedPlot(df, testY)
    # plt.plot(RecallList, PrecisionList, 'ro')
    # plt.axis([0, 1, 0, 1])
    # plt.show()

if __name__ == '__main__':
    main()
