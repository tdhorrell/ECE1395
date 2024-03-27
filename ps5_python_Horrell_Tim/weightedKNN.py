#weightedKNN.py

import numpy as np
from scipy.spatial import distance
import pandas as p

def weightFunction(dist, sigma):
    return np.exp(-(dist ** 2) / sigma ** 2)

def weightedKNN(x_train, y_train, x_test, sigma):
    #define output and intermediaries
    m = np.shape(x_train)
    class_count = 3
    y_test_pred = np.zeros(len(x_test))
    weight_vote = np.zeros((len(x_test), class_count))

    #calculate distances of training data from x_test
    dist = distance.cdist(x_test, x_train, 'euclidean')

    #loop through each test value to add weight
    for i in range(len(x_test)):

        #loop through each training sample to determine ID
        for j in range(len(y_train)):
            #class = 1
            if(y_train[j] == 1):
                weight_vote[i][0] += weightFunction(dist[i][j], sigma)
            if(y_train[j] == 2):
                weight_vote[i][1] += weightFunction(dist[i][j], sigma)
            if(y_train[j] == 3):
                weight_vote[i][2] += weightFunction(dist[i][j], sigma)
    
    for i in range(len(y_test_pred)):
        y_test_pred[i] = (p.Series(weight_vote[i, :]).idxmax() + 1)

    return y_test_pred