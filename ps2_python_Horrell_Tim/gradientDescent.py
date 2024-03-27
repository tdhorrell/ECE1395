#gradientDescent.py

from computeCost import computeCost 
import random

#functions
#helper function partialSum
#returs the summation portion from partial derivites during gradient descent
def partialSum(x, y, theta, feature):
    partialSum = 0

    #loop through rows in x
    for i in range(len(x)):

        #loop through features in x
        for j in range(len(theta)):
            #calculate h(x)
            partialSum += theta[j] * x[i][j]

        #subtract y
        partialSum -= y[i]

        #multiply by the feature itself
        partialSum *= x[i][feature]

    #return the partial sum
    return partialSum


#main function gradientDescent
#returns: theta, cost
def gradientDescent(x_train, y_train, alpha, iters):
    #initialize theta and cost
    theta = []
    cost = [0] * iters
    
    featureSize = x_train.shape

    #randomly initialize theta for features in x with range{-1, 1}
    for i in range(featureSize[1]):
        #theta will explode if completely random in problem 4, theta is set to 0 so the algorithm is stable
        #theta.append(random.random()*2 - 1)
        theta.append(0)
        
    #iterate through gradient descent algorithm
    for i in range(iters):
        #make a temporary array to hold theta values before updating
        tempTheta = [0] * len(theta)

        #loop through each feature
        for j in range(len(theta)):
            tempTheta[j] = (alpha/len(y_train)) * partialSum(x_train, y_train, theta, j)
            
        #loop through theta again to simultaneously update
        for k in range(len(theta)):
            theta[k] -= tempTheta[k]
        
        cost[i] = computeCost(x_train, y_train, theta)

    return theta, cost