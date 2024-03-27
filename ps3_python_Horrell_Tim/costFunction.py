#costFunction.py

import numpy as np
from sigmoid import sigmoid

def costFunction(theta, x_train, y_train):
    m = x_train.shape

    #array for sigmoid function output
    x_trainTheta = np.dot(x_train, theta)

    sigOut = sigmoid(x_trainTheta)

    temp1 = 0

    for i in range(len(sigOut)):
        temp1 += -y_train[i] * np.log(sigOut[i]) - (1 - y_train[i]) * np.log(1 - sigOut[i])
    
    return temp1 / m[0]