#ps3.py

#import functions
import numpy as np
import matplotlib.pyplot as plt
import random
import csv
from scipy.optimize import fmin_bfgs
from sigmoid import sigmoid
from costFunction import costFunction
from gradFunction import gradFunction
from normalEqn import normalEqn

#---------------------------------------
#           problem 1
#---------------------------------------

#load data from the text file
studentData = np.loadtxt('input/hw3_data1.txt', delimiter = ',')

#organize the data
#vectorize the data
testData = np.vstack(studentData[:,0:2])
admittanceData = np.vstack(studentData[:,2])

#find shape of the test data to append ones as a bias term
m = testData.shape
bias = np.ones((m[0], 1))
testData = np.append(bias, testData, axis = 1)

#plot the data as a scatterplot
#first make a color list for admittance
studentCol = np.hstack(np.where(admittanceData < 1, 'r', 'g'))

plt.scatter(testData[:,1], testData[:,2], c = studentCol, s = 10)
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.savefig('output/ps3-1-b.png')


#divide the data into testing and training
#randomize the data using shuffle with them zipped together so they shuffle equally
randomize = list(zip(testData, admittanceData))
random.shuffle(randomize)
res1, res2 = zip(*randomize)
xData, yData = np.array(list(res1)), np.array(list(res2))

#split the data into training and testing sets (90 in training, 10 in testing)
x_train = testData[0:90, 0:3]
x_test = testData[90:100, 0:3]
y_train = admittanceData[0:90]
y_test = testData[90:100]

#test sigmoid function
z, gz = np.linspace(-15, 15, 100), []
for i in range(len(z)):
    gz.append(sigmoid(z[i]))

'''
#make a new plot
plt.plot(z, gz)
plt.xlabel('z')
plt.ylabel('g(z)')
plt.savefig('output/ps3-1-c.png')
plt.show()
'''

#initialize toy data set
x_toy = np.array([[1, 1, 0],
                  [1, 1, 3],
                  [1, 3, 1],
                  [1, 3, 4]])
y_toy = np.array([[0],
                  [1],
                  [0],
                  [1]])
theta_toy = np.array([[2],
                      [0],
                      [0]])

#calculate the toy dataset cost and gradient
toyCost = (costFunction(theta_toy, x_toy, y_toy))
toyGrad = (gradFunction(theta_toy, x_toy, y_toy))
print(toyCost)

#squeeze the array to make y one dimension
y_train = np.squeeze(y_train)

#minimize the cost function using fmin_bfgs
testArgs = (x_train, y_train)
initialTheta = np.array([[0],
                         [0],
                         [0]])

#minimize theta
minTheta = fmin_bfgs(costFunction, initialTheta, gradFunction, args = testArgs)
print(minTheta)

#line of best fit
#y = exam 2, x = exam 1
xSpace = np.linspace(30, 100, 90)
ySpace = -(minTheta[1]*xSpace + minTheta[0])/minTheta[2]
plt.plot(xSpace, ySpace)
plt.savefig('output/ps3-1-f.png')
plt.show()

#calculate percentage of accurate points
#theta1(test1) + theta2(test2) <> 25.07
count = 0
for i in range(len(x_test)):
    if(minTheta[0] + minTheta[1]*x_test[i][1] + minTheta[2]*x_test[i][2] > 0):
        count += 1

#divide count by the total number of tests
accuracy = count / len(x_test)

#prediction for student test1 = 70; test2 = 55
prediction = 0
if(minTheta[0] + minTheta[1]*70 + minTheta[2]*55 > 0):
    prediction = 1

#probability = h(x)
probability = sigmoid(minTheta[0] + minTheta[1]*70 + minTheta[2]*55)

print(probability)
print(accuracy)
print(prediction)

#---------------------------------------
#           problem 2
#---------------------------------------

'''
#read in the data set
data = []

with open('input/hw3_data2.csv', 'r') as file:
    readIn = csv.reader(file, delimiter = ',')

    for row in readIn:
        data.append(row)

#cast data to floats for plotting
floatData = np.array(data).astype(float)
m = floatData.shape

#split the data
population, profit_train = np.reshape(floatData[:,0], (m[0],1)), np.reshape(floatData[:,1], (m[0],1))

#add a column of ones to the X data
ones = np.ones((m[0],1))
popTemp = np.append(ones, population, axis = 1)
pop_train = np.append(popTemp, population ** 2, axis = 1)

utilTheta = normalEqn(pop_train, profit_train)
print(utilTheta)

#build line of best fit arrays
popSpace = np.linspace(500,1000,100)
profitSpace = utilTheta[0][0] + utilTheta[0][1]*popSpace + utilTheta[0][2]*popSpace**2

#generate plot
plt.scatter(floatData[:,0], floatData[:,1], marker = 'x')
plt.plot(popSpace, profitSpace, 'r')
plt.xlabel("population in thousands, n")
plt.ylabel("profit")
plt.savefig('output/ps3-2-b.png')
plt.show()
'''