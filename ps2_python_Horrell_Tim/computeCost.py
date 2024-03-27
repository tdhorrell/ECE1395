#computeCost.py

def computeCost(x, y, theta):
    #initialize two empty variables
    cost = temp1 = 0

    #loop through rows in x
    for i in range(len(x)):

        #loop through features in x
        for j in range(len(theta)):
            #calculate h(x)
            temp1 += theta[j] * x[i][j]

        #sum the running cost according to the cost function
        cost += (temp1 - y[i]) ** 2

        #reset the temp h(x) calculation to zero
        temp1 = 0

    #divide the cost summation by 2m
    return cost/(2*len(x))