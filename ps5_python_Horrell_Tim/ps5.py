#ps5.py

import scipy.io as sio
import sklearn.metrics
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.sparse.linalg import eigs
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from weightedKNN import weightedKNN

#---------------------------------------------------------
#                   Problem 1
#---------------------------------------------------------

'''
data1 = sio.loadmat('input/hw4_data3.mat')

#find keys from data3 and load data
#print(data3.keys())
x_test = np.array(data1["X_test"])
x_train = np.array(data1["X_train"])
y_test = np.array(data1["y_test"])
y_train = np.array(data1["y_train"])
sigma = [0.01, 0.05, 0.2, 1.5, 3.2, 5.0]
accuracy = []

for val in sigma:
    y_test_pred = weightedKNN(x_train, y_train, x_test, val)
    accuracy.append(sklearn.metrics.accuracy_score(y_test, y_test_pred))

print(accuracy)
'''

#---------------------------------------------------------
#                   Problem 2
#---------------------------------------------------------

#first call ps5_split folders to split up data

#create directory paths for training and testing data
testDir = 'input\\test'
trainDir = 'input\\train'

#create an empty array and counter to loop through columns
training_image_array = []
image_counter = 0

#for each training image, vectorize it and fill in the 
for filename in os.listdir(trainDir):
    image_path = os.path.join(trainDir, filename)
    image = Image.open(image_path)
    image_vector = np.reshape(np.asarray(image), (10304, 1), order='F')
    training_image_array.append(image_vector)
    
#get the array into 2d in the correct dimensions
training_image_array = np.array(training_image_array)
training_image_array = np.transpose(training_image_array[:, :, 0])

#training array image
#training_array_display = Image.fromarray(training_image_array, 'L')
#training_array_display.save('output\ps5-2-1-a.png')

#average face vector
average_face_vector = np.transpose(np.mean(training_image_array, axis=1))
average_face_vector = np.reshape(average_face_vector, (10304, 1))
average_face_reconstruct = np.reshape(average_face_vector, (112, 92), order='F')
average_face_reconstruct = average_face_reconstruct.astype(np.uint8)
#average_face_image = Image.fromarray(average_face_reconstruct, 'L')
#average_face_image.save('output\ps5-2-1-b.png')

#calculate covariance matrix
#first find centered data matrix
center_matrix = training_image_array - average_face_vector
covariance_matrix = np.dot(center_matrix, np.transpose(center_matrix))
#covariance_matrix_image = Image.fromarray(covariance_matrix, 'L')
#covariance_matrix_image.save('output\ps5-2-1-c.png')

#compute eigenvalues and sort in descending order

eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
print(eigenvalues.shape)
print(np.shape(covariance_matrix))
eigenvalues_sort = eigenvalues[::-1].sort()

eigenvalues = abs(eigenvalues)

#hold values in lists
k = v_k = []
total_var = sum(eigenvalues)
temp_var = val_count = 0

#loop through each eigenvale
for i in range(len(eigenvalues)):
    #Kth largest eigenvector
    k.append(i+1)

    #add the eigenvalue and get percentage when adding to v(k)
    temp_var += eigenvalues[i]
    v_k.append(temp_var/total_var)
    
    #if v_k is less that 95, update val_count by one
    if(temp_var/total_var < 0.95):
        val_count += 1

#generate plot and save figure
plt.plot(k, v_k)
plt.xlabel('k')
plt.ylabel('v(k)')
plt.savefig('output\ps5-2-1-d.png')

#get 160 largest eigenvalues and eigenvectors
kEigVals, kEigVecs = eigs(covariance_matrix, 160, which = 'LM')
kEigVecs = np.abs(kEigVecs)

#create a big image with all 9 faces
top_eigen_faces = []
for i in range(9):
    #rehspae the eigenvector into an image
    eigen_face = np.reshape(kEigVecs[:,i], (112, 92), order='F')
    #standardize data for image output
    eigen_face = 255*preprocessing.scale(abs(eigen_face))
    top_eigen_faces.append(eigen_face)

#convert to a np array and stack each of the top 9 images
top_eigen_faces = np.array(top_eigen_faces)
top_eigen_faces = np.resize(top_eigen_faces, (1008, 92))

#save the image to output
eigen_face_image = Image.fromarray(top_eigen_faces.astype(np.uint8), 'L')
eigen_face_image.save('output\ps5-2-1-e.png')

#create a matrix W_training where each row corresponds to one reduced training image
#start by taking transpose of U and making empty vectors
kEigVecsTp = np.transpose(kEigVecs)
W_train = []
y_train = []
W_test = []
y_test = []

#begin loop
for filename in os.listdir(trainDir):
    image_path = os.path.join(trainDir, filename)
    image = Image.open(image_path)
    image_vector = np.reshape(np.asarray(image), (10304, 1), order='F')
    image_minus_mean = image_vector - average_face_vector
    W_train.append(np.dot(kEigVecsTp, image_minus_mean))
    y_train.append(int(filename[0]))

for filename in os.listdir(testDir):
    image_path = os.path.join(testDir, filename)
    image = Image.open(image_path)
    image_vector = np.reshape(np.asarray(image), (10304, 1), order='F')
    image_minus_mean = image_vector - average_face_vector
    W_test.append(np.dot(kEigVecsTp, image_minus_mean))
    y_test.append(int(filename[0]))

#reduce dimensionality of lists
W_train = np.array(W_train)
W_test = np.array(W_test)
W_train = W_train[:, :, 0]
W_test = W_test[:, :, 0]

#train a KNN algorithm of varying k values
nearest = np.arange(1, 12, 2)
knn_test_acc = np.empty(len(nearest))

#loop through and compute accuracy for each example
for i, k in enumerate(nearest):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(W_train, y_train)

    knn_test_acc[i] = knn.score(W_test, y_test)

#train SVM classifiers starting with one vs one
#start time
train_time = []
test_time = []
svm_test_accuracy = []

#linear ovo
startTime = time.time()
svm_linear_ovo = SVC(decision_function_shape='ovo', kernel='linear').fit(W_train, y_train)
endTime = time.time()
train_time.append(endTime - startTime)

startTime = time.time()
prediction = svm_linear_ovo.predict(W_test)
endTime = time.time()
test_time.append(endTime - startTime)
svm_test_accuracy.append(sklearn.metrics.accuracy_score(y_test, prediction))

#poly ovo
startTime = time.time()
svm_poly_ovo = SVC(decision_function_shape='ovo', kernel='poly').fit(W_train, y_train)
endTime = time.time()
train_time.append(endTime - startTime)

startTime = time.time()
prediction = svm_poly_ovo.predict(W_test)
endTime = time.time()
test_time.append(endTime - startTime)
svm_test_accuracy.append(sklearn.metrics.accuracy_score(y_test, prediction))

#rbf ovo
startTime = time.time()
svm_rbf_ovo = SVC(decision_function_shape='ovo', kernel='rbf').fit(W_train, y_train)
endTime = time.time()
train_time.append(endTime - startTime)

startTime = time.time()
prediction = svm_rbf_ovo.predict(W_test)
endTime = time.time()
test_time.append(endTime - startTime)
svm_test_accuracy.append(sklearn.metrics.accuracy_score(y_test, prediction))

#linear ovr
startTime = time.time()
svm_linear_ovr = SVC(decision_function_shape='ovr', kernel='linear').fit(W_train, y_train)
endTime = time.time()
train_time.append(endTime - startTime)

startTime = time.time()
prediction = svm_linear_ovr.predict(W_test)
endTime = time.time()
test_time.append(endTime - startTime)
svm_test_accuracy.append(sklearn.metrics.accuracy_score(y_test, prediction))

#poly ovr
startTime = time.time()
svm_poly_ovr = SVC(decision_function_shape='ovr', kernel='poly').fit(W_train, y_train)
endTime = time.time()
train_time.append(endTime - startTime)

startTime = time.time()
prediction = svm_poly_ovr.predict(W_test)
endTime = time.time()
test_time.append(endTime - startTime)
svm_test_accuracy.append(sklearn.metrics.accuracy_score(y_test, prediction))

#rbf ovr
startTime = time.time()
svm_rbf_ovr = SVC(decision_function_shape='ovr', kernel='rbf').fit(W_train, y_train)
endTime = time.time()
train_time.append(endTime - startTime)

startTime = time.time()
prediction = svm_rbf_ovr.predict(W_test)
endTime = time.time()
test_time.append(endTime - startTime)
svm_test_accuracy.append(sklearn.metrics.accuracy_score(y_test, prediction))

print(train_time)
print(test_time)
print(svm_test_accuracy)