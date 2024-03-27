#ps7.py
import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from predict import predict
from nnCost import nnCost
from sGD import sGD
from sigmoidGradient import sigmoidGradient
from sklearn.metrics import accuracy_score

#---------------------------------------------------------
#                   Problem 0
#---------------------------------------------------------

#load in data from the iris set
#import all the data
iris_data = sio.loadmat('input/HW7_Data.mat')
iris_weights = sio.loadmat('input/HW7_weights_2.mat')

#get keys
print(iris_data.keys())
print(iris_weights.keys())

#split data and thetas
iris_X = np.array(iris_data['X'])
iris_y = np.array(iris_data['y'])
iris_theta1 = np.array(iris_weights['Theta1'])
iris_theta2 = np.array(iris_weights['Theta2'])

#verify dimensions
print(f'Dimensions of iris_X: {iris_X.shape}')
print(f'Dimensions of iris_y: {iris_y.shape}')
print(f'Dimensions of theta1: {iris_theta1.shape}')
print(f'Dimensions of theta2: {iris_theta2.shape}')

#turn y values from 1-3 to vectors [0, 0, 0]
iris_y_vector = []
for output in iris_y:
    if output == 1:
        iris_y_vector.append([1, 0, 0])
    if output == 2:
        iris_y_vector.append([0, 1, 0])
    if output == 3:
        iris_y_vector.append([0, 0, 1])

#override iris_y
iris_y = np.array(iris_y_vector)
print(f'Dimensions of updated iris_y: {iris_y.shape}')

#---------------------------------------------------------
#                   Problem 1
#---------------------------------------------------------

#call the prediction function to test accuracy
p, h_x = predict(iris_theta1, iris_theta2, iris_X)

print(f'Prediction accuracy: ',accuracy_score(p, iris_y),'%')

#---------------------------------------------------------
#                   Problem 2
#---------------------------------------------------------

#calculate the cost given multiple values of lambda
print(f'Cost of lambda = 0: ',nnCost(iris_theta1, iris_theta2, iris_X, iris_y, 3, 0))
print(f'Cost of lambda = 1: ',nnCost(iris_theta1, iris_theta2, iris_X, iris_y, 3, 1))
print(f'Cost of lambda = 2: ',nnCost(iris_theta1, iris_theta2, iris_X, iris_y, 3, 2))

#---------------------------------------------------------
#                   Problem 3
#---------------------------------------------------------

#print gradient when input is givet
print(f'g_prime: ',sigmoidGradient([-10, 0, 10]))
print('\n')

#---------------------------------------------------------
#                   Problem 4
#---------------------------------------------------------

#save graph from function
theta1, theta2 = sGD(4, 8, 3, iris_X, iris_y, 1, 0.001, 1)

#---------------------------------------------------------
#                   Problem 5
#---------------------------------------------------------

#split the data into testing and training
x_train, x_test, y_train, y_test = train_test_split(iris_X, iris_y, train_size=0.85, random_state=23)

lam_vals = [0, 0.01, 0.1, 1]
epoch_vals = [50, 100]

for epoch in epoch_vals:
    for lam in lam_vals:
        print(f'lambda: ',lam,'    epochs: ',epoch)
        #values for epochs of 50
        theta1, theta2 = sGD(4, 8, 3, x_train, y_train, lam, 0.001, epoch)

        #predict testing and training accuracy
        p_train, h_x_train = predict(theta1, theta2, x_train)
        p_test, h_x_test = predict(theta1, theta2, x_test)

        #get accuracy scores
        print(f'Training accuracy: ',accuracy_score(p_train, y_train)*100,'%')
        print(f'Testing accuracy: ',accuracy_score(p_test, y_test)*100,'%')

        #get cost
        print(f'Training cost: ',nnCost(theta1, theta2, x_train, y_train, 3, lam))
        print(f'Testing cost: ',nnCost(theta1, theta2, x_test, y_test, 3, lam))
        print('\n')
