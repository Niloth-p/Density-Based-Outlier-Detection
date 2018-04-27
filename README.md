# Density-Based-Outlier-Detection
Description
An implementation of a density based outlier detection method - the Local Outlier Factor Technique,
to find frauds in credit card transactions (here) using Python. For detecting both local and global outliers.

Dataset Used:
German Credit Data
Professor Dr. Hans Hofmann  
Institut f"ur Statistik und "Okonometrie  
Universit"at Hamburg  
numerical attributes, Strathclyde University 
produced the file "german.data-numeric".
(1 = Good, 2 = Bad)

How to run:

    After setting the correct values of the global variables
    python Outlier.py

Global variables:
    
    first_time - If run for the first time, the distance matrix is computed after reading the data from the dataset, and is written to a pickle file called "distancematrix". If it is set to false, the distance matrix is directly read from the pickle file.

    filename - The name of the data file

    parameters - a list of pairs of k and O values, for which the outlier detection is done

Parameters: 
   
    k - to get kNN, kdist, and hence LOF (here, it doesn't seem to affect the accuracy!?)
    O - Number of outliers (higher the number of outliers, higher the precision and recall seem to be)
    N - len(data) (= 1000 for the given dataset)

    The Precision averages around 0.62, and the Recall linearly increases with increase in O (!?)

    The plotting of the PR curve has been commented out.

Functions:
def readData():
    
    Reads data from text file and stores as data frame using pandas.
    df(X) is taken as the entire table except the last column containing the classification of points as outliers or not
    And the last column is taken as Y for measuring the accuracy of the training
    The data is normalized

def mahalanobisdist(a, b):
    
    Calculates the mahalanobis distance between 2 points of the data
    d(x,y) = sqrt((x-y)T . S^-1 . (x-y))
    Sinv, the inverse of the covariance matrix is computed without issue of singularity arising, using pinv()
    
def createDistanceMatrix(data, first_timeval, N):
    
    Computes the distance matrix (the Mahalanobis distance between all pairs of points) and 
    writes to to a pickle file to save time on future runs, which is indicated by the global variable first_time
    
def getLRD(N, distancematrix, k, data):
    
    Finds, for each point,
    1. The KNN
    2. The kdistance for each point i.e the distance to its kthNN,
    2. The number of points that fall within the k-distance neighbourhood (Nk)
    3. Reachability distances (betw the point in focus and all other points(1 at a time))
    i.e max{k-distance of point in focus, distance between point in focus and the other point in consideration}
    4. LRD (local reachability density)
    i.e Nk/(sum of reachability distances)
    Lower the LRD, higher the LOF => considering the LRD value for detecting outliers
    
def getReachabilityDistances(N, data, kdist, distancematrix):
    
    Calculates the reachability distance between all pairs of points
    
def getAccuracy(outliers, Y, N, PrecisionList, RecallList):
    
    Gets the performance metrics of the outlier detection done,
    in terms of Accuracy, Precision, Recall, F1-Score
    using true and false +ves and -ves, by comparing the obtained classification vs the given good and bad points
    True +ve : Good point classified good
    True -ve : Bad point classified bad
    False +ve : Bad point classified good
    False -ve : Good point classified bad
    All 4 of these types are not equally important. 
    For example, false -ves are more acceptable than false +ves.
    Accuracy = (tp + tn)/(tp + tn + fp + fn)
    Precision = tp/(tp + fp)
    Recall = tp/(tp + fn)
    F1 = HM of Precision and Recall

def main():
    
    Calls the functions 
    1.to get distance matrix,
    2.the LRD, 
    3.the 1st O points after sorting of LRD and 
    4.gets the performance metrics
    
