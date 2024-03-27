#gradFunction.py

import numpy as np
from sigmoid import sigmoid

def gradFunction(theta, x_train, y_train):
    #initialize and intermediaries
    m = x_train.shape
    x_trainT = np.transpose(x_train)

    #will produce a column with length x_train, width 1
    x_trainTheta = np.dot(x_train, theta)
    
    #subtract y values from h(x), then take the dot product do get a (3x1) output matrix with gradient
    temp = np.dot(x_trainT, sigmoid(x_trainTheta) - y_train)

    #calculate the average by dividing by the number of samples
    #flatten into a 1D array
    return np.ndarray.flatten(1/m[0] * temp)