#nnCost.py
import numpy as np
from predict import predict

def nnCost(Theta1, Theta2, X, y, K, lam):
    #get h_x as the prediction
    p, h_x = predict(Theta1, Theta2, X)

    m = X.shape[0]

    #initialize the prediction cost
    pred_cost = 0

    #loop through to find prediction cost
    for i in range(m):
        for k in range(K):
            pred_cost += (y[i][k] * np.log(h_x[i][k])) + ((1 - y[i][k]) * np.log(1 - h_x[i][k]))

    #find cost of theta1 and theta2
    theta1_cost = np.sum(np.square(Theta1))
    theta2_cost = np.sum(np.square(Theta2))

    #return J(theta)
    return ((-1/m) * pred_cost) + ((lam/(2 * m)) * (theta1_cost + theta2_cost))