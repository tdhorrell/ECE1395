#ps2.py

#import functions from all other files
from computeCost import computeCost
from gradientDescent import gradientDescent
from normalEqn import normalEqn
import numpy as np
import csv
import matplotlib.pyplot as plt
import random

#---------------------------------------
#           problems 1-3
#---------------------------------------

'''
#initialize toy data set for computeCost
x = np.array([[1, 0],
             [1, 2],
             [1, 3],
             [1, 4]])
y = np.array([[4],
              [8],
              [10],
              [12]])

theta1 = [0, 0.5]
theta2 = [1, 1]

#find costs
print(computeCost(x, y, theta1))
print(computeCost(x, y, theta2))

#do gradient descent
theta, cost = gradientDescent(x, y, 0.001, 15)

#save variables in np arrays to print output without 'array' text
theta = np.hstack(theta)
cost = np.hstack(cost)
print(theta)
print(cost)


thetaNormal = normalEqn(x, y)
print(thetaNormal)
'''

#---------------------------------------
#           problem 4
#---------------------------------------


#load data into python
data = []

with open('Input/hw2_data1.csv', 'r') as file:
    readIn = csv.reader(file, delimiter = ',')

    for row in readIn:
        data.append(row)

#cast data to floats for plotting
floatData = np.array(data).astype(float)

#plot scatterplot
plt.scatter(floatData[:,0], floatData[:,1], marker = 'X', color='r', linewidths = 0.25)
plt.xlim(0,3)
plt.ylim(5,50)
plt.xlabel("Horse power of a car in 100s")
plt.ylabel("Price in $1,000s")
plt.savefig('Output/ps2-4-b.png')
#plt.show()

#add a column of ones to the X data
m = floatData.shape
ones = np.ones((m[0],1))
fullData = np.append(ones, floatData, axis = 1)

#randomize the data using a seed to keep the same data split
random.seed(1)
random.shuffle(fullData)

#split the data into training and testing sets (179*.9 = 161 data points in the training set, 18 in the testing set)
x_train = fullData[0:161, 0:2]
x_test = fullData[161:180, 0:2]
y_train = fullData[0:161, 2]
y_test = fullData[161:180, 2]

#do gradient descent on the data
carTheta, carCost = gradientDescent(x_train, y_train, 0.3, 500)

#save data as numpy arrays
carTheta = np.vstack(carTheta)
carCost = np.vstack(carCost)

#plot the line of best fit
xSpace = np.linspace(0, 3, 30)
plt.plot(xSpace, carTheta[0] + carTheta[1]*xSpace)
plt.legend(['test data' , 'theta0 + theta1 * x'], loc = 'upper left')
plt.savefig('Output/ps2-4-e-2.png')
plt.show()

#plot the cost vs iterations
plt.plot(carCost)
plt.xlabel("# of iterations")
plt.ylabel("Cost")
plt.title("alpha = 0.3")
plt.savefig('Output/ps2-4-h-4.png')
plt.show()

#compute mean square error
meanSquaredError = 0
y_pred = [0] * len(y_test)

#build y_pred and sum the square error
for i in range(len(y_test)):
    y_pred[i] = (carTheta[0]*x_test[i][0] + carTheta[1]*x_test[i][1])
    meanSquaredError += (y_pred[i] - y_test[i]) ** 2

#divide total square error by the length to get the meanSquaredError
meanSquaredError /= len(y_test)
print("Mean Squared Error is: " + str(meanSquaredError))

normalTheta = normalEqn(x_train, y_train)
print("Normal Theta: " + str(normalTheta))

#compute mean square error for the normal equation
#update y_pred and sum the square error
for i in range(len(y_test)):
    y_pred[i] = (normalTheta[0][0]*x_test[i][0] + normalTheta[0][1]*x_test[i][1])
    meanSquaredError += (y_pred[i] - y_test[i]) ** 2

#divide total square error by the length to get the meanSquaredError
meanSquaredError /= len(y_test)
print("Normal Mean Squared Error is: " + str(meanSquaredError))


#---------------------------------------
#           problem 5
#---------------------------------------

#load data from the text file
houseData = np.loadtxt('Input/hw2_data2.txt', delimiter = ',')

#vectorize the data
houseSize = np.vstack(houseData[:,0])
houseBedrooms = np.vstack(houseData[:,1])
houseCost = np.vstack(houseData[:,2])

#regularize the data
houseSizeMean = np.mean(houseSize)
houseSizeStd = np.std(houseSize)
houseSize = (houseSize - houseSizeMean) / houseSizeStd
print("House size mean: " + str(houseSizeMean))
print("House size std: " + str(houseSizeStd))

houseBedroomsMean = np.mean(houseBedrooms)
houseBedroomsStd = np.std(houseBedrooms)
houseBedrooms = (houseBedrooms - houseBedroomsMean) / houseBedroomsStd
print("House bedrooms mean: " + str(houseBedroomsMean))
print("House bedrooms std: " + str(houseBedroomsStd))

#add a column of ones to houseSize and houseBedrooms
m = houseSize.shape
standardData = np.ones((m[0],1))
standardData = np.append(standardData, houseSize, axis = 1)
standardData = np.append(standardData, houseBedrooms, axis = 1)

#do gradient descent
hTheta, hCost  = gradientDescent(standardData, houseCost, 0.01, 750)

'''
#plot the cost vs iterations
plt.plot(np.vstack(hCost))
plt.xlabel("# of iterations")
plt.ylabel("Cost")
plt.title("alpha = 0.01")
plt.savefig('Output/ps2-5-b.png')
plt.show()
'''

#print theta values
print("Theta0: " + str(hTheta[0]))
print("Theta1: " + str(hTheta[1]))
print("Theta2: " + str(hTheta[2]))

