#ps4.py

import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from Reg_normalEqn import Reg_normalEqn
from computeCost import computeCost
from logReg_multi import logReg_multi
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as knc

#import all the data
data1 = sio.loadmat('input/hw4_data1.mat')
data2 = sio.loadmat('input/hw4_data2.mat')
data3 = sio.loadmat('input/hw4_data3.mat')

#---------------------------------------------------------
#                   Problem 1
#---------------------------------------------------------

#find keys from data1
print(data1.keys())
data1_x = np.array(data1["X_data"])
data1_y = np.array(data1["y"])

#store lambda values in an array and hold average cost values
lamb = np.array([0, 0.001, 0.003, 0.005, 0.007, 0.009, 0.012, 0.017])
trainCost = testCost = np.zeros(8)

#use a for loop to loop through data 20 times
for i in range(20):

    #seperate the data using sklearn
    x_train, x_test, y_train, y_test = train_test_split(data1_x, data1_y, train_size=0.88, random_state=i)
    
    #use another loop to iterate through each lambda
    for j in range(len(lamb)):
        theta = Reg_normalEqn(x_train, y_train, lamb[j])
        trainCost[j] += computeCost(x_train, y_train, theta)
        testCost[j] += computeCost(x_test, y_test, theta)

#divide each cost by the number of lambda tests
trainCost[:] = [x / len(lamb) for x in trainCost]
testCost[:] = [x / len(lamb) for x in testCost]

plt.plot(lamb, trainCost, '-bo')
plt.plot(lamb, testCost, '-ro')
plt.legend(["Training Error", "Testing Error"])
plt.xlabel("Values of lambda")
plt.ylabel("Average Error")
plt.savefig("output/ps4-1-a.png")
plt.show()


#---------------------------------------------------------
#                   Problem 2
#---------------------------------------------------------

'''
#find keys from data2 and load data
print(data2.keys())
x1Data = np.array(data2["X1"])
x2Data = np.array(data2["X2"])
x3Data = np.array(data2["X3"])
x4Data = np.array(data2["X4"])
x5Data = np.array(data2["X5"])
y1Data = np.array(data2["y1"])
y2Data = np.array(data2["y2"])
y3Data = np.array(data2["y3"])
y4Data = np.array(data2["y4"])
y5Data = np.array(data2["y5"])

#build each of the five classifiers
#classifier 1
x_train1 = np.vstack((x1Data, x2Data, x3Data, x4Data))
y_train1 = np.squeeze(np.vstack((y1Data, y2Data, y3Data, y4Data)))
x_test1 = x5Data
y_test1 = np.squeeze(y5Data)
#classifier 2
x_train2 = np.vstack((x1Data, x2Data, x3Data, x5Data))
y_train2 = np.squeeze(np.vstack((y1Data, y2Data, y3Data, y5Data)))
x_test2 = x4Data
y_test2 = np.squeeze(y4Data)
#classifier 3
x_train3 = np.vstack((x1Data, x2Data, x4Data, x5Data))
y_train3 = np.squeeze(np.vstack((y1Data, y2Data, y4Data, y5Data)))
x_test3 = x3Data
y_test3 = np.squeeze(y3Data)
#classifier 4
x_train4 = np.vstack((x1Data, x3Data, x4Data, x5Data))
y_train4 = np.squeeze(np.vstack((y1Data, y3Data, y4Data, y5Data)))
x_test4 = x2Data
y_test4 = np.squeeze(y2Data)
#classifier 5
x_train5 = np.vstack((x2Data, x3Data, x4Data, x5Data))
y_train5 = np.squeeze(np.vstack((y2Data, y3Data, y4Data, y5Data)))
x_test5 = x1Data
y_test5 = np.squeeze(y1Data)

#save k values to iterate through
kVal = np.array([x for x in range (1, 15, 2)])
accuracy = np.zeros(len(kVal))

#loop through each value of k
for i in range(len(kVal)):
    #run knn algorithm at k nearest neighbors and fit the data
    neigh = knc(kVal[i])
    neigh.fit(x_train1, y_train1)
    accuracy[i] += neigh.score(x_test1, y_test1)

    neigh.fit(x_train2, y_train2)
    accuracy[i] += neigh.score(x_test2, y_test2)

    neigh.fit(x_train3, y_train3)
    accuracy[i] += neigh.score(x_test3, y_test3)

    neigh.fit(x_train4, y_train4)
    accuracy[i] += neigh.score(x_test4, y_test4)

    neigh.fit(x_train5, y_train5)
    accuracy[i] += neigh.score(x_test5, y_test5)    

    accuracy[i] /= 5

plt.plot(kVal, accuracy, '-bo')
plt.xlabel("K Nearest Neighbor")
plt.ylabel("Average Accuracy")
plt.savefig('output/ps4-2-a.png')
plt.show()
'''

#---------------------------------------------------------
#                   Problem 3
#---------------------------------------------------------


#find keys from data3 and load data
#print(data3.keys())
x_test = np.array(data3["X_test"])
x_train = np.array(data3["X_train"])
y_test = np.array(data3["y_test"])
y_train = np.array(data3["y_train"])

testPred = logReg_multi(x_train, y_train, x_test)
trainPred = logReg_multi(x_train, y_train, x_train)

#calculate test accuracy
testCount = 0
for i in range(len(y_test)):
    if(y_test[i] == testPred[i]):
        testCount += 1
testAccuracy = testCount / len(y_test)

#calculate training accuracy
trainCount = 0
for i in range(len(y_train)):
    if(y_train[i] == trainPred[i]):
        trainCount += 1
trainAccuracy = trainCount / len(y_train)

print("Test Acc: " + str(testAccuracy))
print("Train Acc: " + str(trainAccuracy))
