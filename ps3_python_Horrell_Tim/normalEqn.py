#normalEqn.py

import numpy as np

#returns theta
def normalEqn(x_train, y_train):
    #fill x and y with the training data
    x, y = np.array(x_train), np.vstack(y_train)



    '''
    #ONLY NEEDED IN ORIGINAL IMPLEMENTATION TO SHAPE AND ADD ONES
    #create an array of ones to add for bias x0
    ones = np.ones((m,1))

    #reshape x to be two dimensional and vertical then add bias
    x = np.reshape(x, (m,1))
    xBias = np.append(ones, x, axis = 1)
    '''
    
    #calculate intermediaries for normal equation
    xT = np.transpose(x)
    
    xTx = np.linalg.pinv(xT.dot(x))
    xTy = xT.dot(y)

    theta = np.transpose(xTx.dot(xTy))

    return theta