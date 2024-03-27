#normalEqn.py

import numpy as np

#returns theta
def Reg_normalEqn(x_train, y_train, lamb):
    #fill x and y with the training data
    x, y = np.array(x_train), np.vstack(y_train)

    #calculate the diagonal matrix and set bias feature to 0
    m = x.shape
    diag = np.eye(m[1], m[1])
    diag[0][0] = 0

    #calculate intermediaries for normal equation
    xT = np.transpose(x)
    lamD = np.dot(lamb,diag)
    xTy = xT.dot(y)
    xT = np.transpose(x)
    temp1 = np.linalg.pinv(xT.dot(x) + lamD)

    theta = temp1.dot(xTy)

    return theta