#computeCost.py

import numpy as np

def computeCost(x, y, theta):
    m = x.shape
    cost = 0
    theta = np.array(theta)

    #hypothesis values for x
    xTheta = np.dot(x, theta)

    xThetaOut = np.zeros(len(xTheta))

    #The i is each ROW
    for i in range(len(xTheta)):
        xThetaOut[i] = np.sum(np.multiply(theta, xTheta[i]))

    for i in range(m[0]):
        error = (y[i] - xThetaOut[i]) ** 2
        cost += error
    
    return cost / (2*m[0])