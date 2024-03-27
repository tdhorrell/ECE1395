#logReg_multi.py

import numpy as np
from sklearn.linear_model import LogisticRegression

def logReg_multi(x_train, y_train, x_test):
    #initialize an array to hold the all vs one data
    y_train_all = np.zeros((len(y_train), 3))
    
    #make the training arrays one vs all approach
    for i in range(3):
        for j in range(len(y_train)):
            if(y_train[j] == (i + 1)):
                y_train_all[j][i] = 1
            else:
                y_train_all[j][i] = 0

    #call the model for each training set
    logisticReg1 = LogisticRegression(random_state=0).fit(x_train, y_train_all[:, 0])
    logisticReg2 = LogisticRegression(random_state=0).fit(x_train, y_train_all[:, 1])
    logisticReg3 = LogisticRegression(random_state=0).fit(x_train, y_train_all[:, 2])

    #greatest probability prediction. Make a temp probability and temp prediction array
    pred = np.zeros(len(x_test))
    tempP1 = tempP2 = tempP3 = []

    #predict probabilities of each fit
    tempP1 = logisticReg1.predict(x_test)
    tempP2 = logisticReg2.predict(x_test)
    tempP3 = logisticReg3.predict(x_test)

    for i in range(len(x_test)):
        #use conditional statements to determine which output it will be
        if(max(tempP1[i], tempP2[i], tempP3[i]) == tempP1[i]):
            pred[i] = 1
        elif(max(tempP1[i], tempP2[i], tempP3[i]) == tempP2[i]):
            pred[i] = 2
        else:
            pred[i] = 3

    return pred



    



