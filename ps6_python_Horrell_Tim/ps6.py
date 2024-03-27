#ps6.py

import scipy.io as sio
import numpy as np
from numpy.linalg import inv, det
from sklearn.metrics import accuracy_score

#import all the data
data3 = sio.loadmat('input/hw4_data3.mat')

#---------------------------------------------------------
#                   Problem 0
#---------------------------------------------------------

#find keys from data3 and load data
#print(data3.keys())
x_test = np.array(data3["X_test"])
x_train = np.array(data3["X_train"])
y_test = np.array(data3["y_test"])
y_train = np.array(data3["y_train"])

x_train_1 = []
x_train_2 = []
x_train_3 = []

#seperate into classes
for i in range(len(y_train)):
    if y_train[i] == 1:
        x_train_1.append(x_train[i,:])
    if y_train[i] == 2:
        x_train_2.append(x_train[i,:])
    if y_train[i] == 3:
        x_train_3.append(x_train[i,:])

x_train_1 = np.array(x_train_1)
x_train_2 = np.array(x_train_2)
x_train_3 = np.array(x_train_3)

print(x_train_1.shape)
print(x_train_2.shape)
print(x_train_3.shape)

#---------------------------------------------------------
#                   Problem 1
#---------------------------------------------------------

#calculate mean for each feature
x_train_1_mean = np.mean(x_train_1, axis=0)
x_train_2_mean = np.mean(x_train_2, axis=0)
x_train_3_mean = np.mean(x_train_3, axis=0)

#calculate std for each vector
x_train_1_std = np.std(x_train_1, axis=0)
x_train_2_std = np.std(x_train_2, axis=0)
x_train_3_std = np.std(x_train_3, axis=0)

print(x_train_1_mean)
print(x_train_1_std)
print(x_train_2_mean)
print(x_train_2_std)
print(x_train_3_mean)
print(x_train_3_std)

#calculate p(xj|w1), p(xj|w2), p(xj|w3)
pj_w1 = [0] * 4
pj_w2 = [0] * 4
pj_w3 = [0] * 4
for i in range(4):
    pj_w1[i] = 1/(np.sqrt(2*np.pi)*x_train_1_std[i]) * np.exp(-((x_test[:,i] - x_train_1_mean[i]) ** 2) / (2 * x_train_1_std[i]))
    pj_w2[i] = 1/(np.sqrt(2*np.pi)*x_train_2_std[i]) * np.exp(-((x_test[:,i] - x_train_2_mean[i]) ** 2) / (2 * x_train_2_std[i]))
    pj_w3[i] = 1/(np.sqrt(2*np.pi)*x_train_3_std[i]) * np.exp(-((x_test[:,i] - x_train_3_mean[i]) ** 2) / (2 * x_train_3_std[i]))

#calculate ln(p(x|wi))
pj_w1_ln = np.log(np.sum(pj_w1, axis=0))
pj_w2_ln = np.log(np.sum(pj_w2, axis=0))
pj_w3_ln = np.log(np.sum(pj_w3, axis=0))

#calculate posterior probabilities
post_w1 = pj_w1_ln + np.log(1/3)
post_w2 = pj_w2_ln + np.log(1/3)
post_w3 = pj_w3_ln + np.log(1/3)

print("Posterior probabilities:")
print(post_w1)
print(post_w2)
print(post_w3)

#classify samples based on posterior probability
y_test_pred = []
for i in range(len(y_test)):
    if max(post_w1[i], post_w2[i], post_w3[i]) == post_w1[i]:
        y_test_pred.append(1)
    if max(post_w1[i], post_w2[i], post_w3[i]) == post_w2[i]:
        y_test_pred.append(2)
    if max(post_w1[i], post_w2[i], post_w3[i]) == post_w3[i]:
        y_test_pred.append(3)

classifier_acc = accuracy_score(y_test, y_test_pred)
print(classifier_acc)

#---------------------------------------------------------
#                   Problem 2
#---------------------------------------------------------

#get covariance estimate for each class
sigma_1 = np.cov(np.transpose(x_train_1))
sigma_2 = np.cov(np.transpose(x_train_2))
sigma_3 = np.cov(np.transpose(x_train_3))

#print covariance matricies
print(sigma_1.shape)
print(sigma_1)
print(sigma_2.shape)
print(sigma_2)
print(sigma_3.shape)
print(sigma_3)

#mean vectors
print(x_train_1_mean.shape)
print(x_train_1_mean)
print(x_train_2_mean.shape)
print(x_train_2_mean)
print(x_train_3_mean.shape)
print(x_train_3_mean)

#calculate discriminatnt function for each class
g_1 = [0] * len(y_test)
g_2 = [0] * len(y_test)
g_3 = [0] * len(y_test)

test = x_test[0,:] - x_train_1_mean
print(test.shape)

#loop through all testing values
#could find (x - ui) to make quicker
for i in range(len(y_test)):
    g_1[i] = (-1/2) * np.dot(np.dot(np.transpose(x_test[i,:] - x_train_1_mean), inv(sigma_1)), (x_test[i,:] - x_train_1_mean)) + np.log(1/3) + ((-4/2) * np.log(2*np.pi)) + ((-1/2) * np.log(det(sigma_1)))
    g_2[i] = (-1/2) * np.dot(np.dot(np.transpose(x_test[i,:] - x_train_2_mean), inv(sigma_2)), (x_test[i,:] - x_train_2_mean)) + np.log(1/3) + ((-4/2) * np.log(2*np.pi)) + ((-1/2) * np.log(det(sigma_2)))
    g_3[i] = (-1/2) * np.dot(np.dot(np.transpose(x_test[i,:] - x_train_3_mean), inv(sigma_3)), (x_test[i,:] - x_train_3_mean)) + np.log(1/3) + ((-4/2) * np.log(2*np.pi)) + ((-1/2) * np.log(det(sigma_3)))

#classify samples based on posterior probability
y_test_pred_mle = []
for i in range(len(y_test)):
    if max(g_1[i], g_2[i], g_3[i]) == g_1[i]:
        y_test_pred_mle.append(1)
    if max(g_1[i], g_2[i], g_3[i]) == g_2[i]:
        y_test_pred_mle.append(2)
    if max(g_1[i], g_2[i], g_3[i]) == g_3[i]:
        y_test_pred_mle.append(3)

#get accuracy score
classifier_acc_mle = accuracy_score(y_test, y_test_pred_mle)
print(classifier_acc_mle)