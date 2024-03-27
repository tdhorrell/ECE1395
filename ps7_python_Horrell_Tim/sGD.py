#sGD.py
import numpy as np
import matplotlib.pyplot as plt
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient
from nnCost import nnCost

def sGD(input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lam, alpha, max_epochs):
    #define random Theta1 and Theta2
    theta1 = np.random.rand(hidden_layer_size, input_layer_size + 1)
    theta2 = np.random.rand(num_labels, hidden_layer_size + 1)

    #get thetas in range of -0.15 to 0.15
    theta1 = theta1*0.3 - 0.15
    theta2 = theta2*0.3 - 0.15

    #get cost list
    cost = []

    #number of epoch
    for epoch in range(max_epochs):
        #loop through each training example
        for i, q in enumerate(X_train):
            #get the prediction, first format x into columns and add bias
            a1 = np.reshape(np.transpose(q), (4, 1))
            a1_bias = np.vstack([[1], a1])

            #calculate the 2nd activation layer a2
            a2 = sigmoid(np.dot(theta1, a1_bias))
            a2_bias = np.vstack([[1], a2])

            #calculate output layer
            a3 = sigmoid(np.dot(theta2, a2_bias))

            #calculate error term for each layer
            error_term_3 = np.subtract(a3, np.reshape(y_train[i][:], (3,1)))
            error_term_2 = np.multiply(np.delete(np.dot(np.transpose(theta2), error_term_3), 0, 0), sigmoidGradient(np.dot(theta1, a1_bias)))

            #calculate delta terms
            delta_1 = np.dot(error_term_2, np.transpose(a1_bias))
            delta_2 = np.dot(error_term_3, np.transpose(a2_bias))

            #create D matricies for weight updates
            Delta_1 = delta_1
            Delta_2 = delta_2

            for i in range(1, Delta_1.shape[1]):
                Delta_1[:,i] = delta_1[:,i] + lam*theta1[:,i]

            for i in range(1, Delta_2.shape[1]):
                Delta_2[:,i] = delta_2[:,i] + lam*theta2[:,i]

            theta1 = np.subtract(theta1, alpha * Delta_1)
            theta2 = np.subtract(theta2, alpha * Delta_2)

            #calculate cost
            cost.append(nnCost(theta1, theta2, X_train, y_train, 3, lam))

        #plot the cost per stochastic iteration
        #plt.plot(cost)
        #plt.xlabel('iteration')
        #plt.ylabel('cost')
        #plt.savefig('output/ps7-e-1.png')

    return theta1, theta2
