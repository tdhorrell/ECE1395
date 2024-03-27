#predict.py
import numpy as np
from sigmoid import sigmoid

#function to predict output of neural networks
def predict(Theta1, Theta2, X):
    #format x into columns
    X = np.transpose(X)

    #add a new row of ones as a bias term
    X = np.vstack([np.ones(X.shape[1]), X])

    #calculate the 2nd activation layer a2
    a2 = sigmoid(np.dot(Theta1, X))

    #add a bias term to a2
    a2 = np.vstack([np.ones(a2.shape[1]), a2])

    #calculate output layer
    a3 = sigmoid(np.dot(Theta2, a2))

    #get a3 into form for h_x
    h_x = np.transpose(a3)

    #calculate a prediction
    p = []
    for val in h_x:
        if(max(val) == val[0]):
            p.append([1, 0, 0])
        if(max(val) == val[1]):
            p.append([0, 1, 0])
        if(max(val) == val[2]):
            p.append([0, 0, 1])

    return p, h_x
