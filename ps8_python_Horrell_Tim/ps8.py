# ps8.py
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import stats as st
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from kmeans_single import kmeans_single
from kmeans_multiple import kmeans_multiple

#helper function to print testing error
def class_error(mnist_fold_X, mnist_fold_y, clf, model_type):
    for i, fold in enumerate(mnist_fold_X):
        pred = clf.predict(fold)
        if(i == 5):
            print(f'Prediction error of ',model_type,' Testing: ',(1 - accuracy_score(pred, mnist_fold_y[i]))*100,'%\n')
        else:
            print(f'Prediction error of ',model_type,' Fold ',(i+1),': ',(1 - accuracy_score(pred, mnist_fold_y[i]))*100,'%')

#---------------------------------------------------------
#                   Problem 1
#---------------------------------------------------------

'''
#load data
mnist_data = sio.loadmat('input/HW8_data1.mat')
#print(mnist_data.keys())
mnist_x = np.array(mnist_data['X'])
mnist_y = np.array(mnist_data['y'])

#load numpy rng and get random images
rng = np.random.default_rng()
image_num = rng.integers(0, 4999, 25)

#fill the image list with reshaped images
im_list = []
for i, num in enumerate(image_num.T):
    im_list.append(np.reshape(mnist_x[num,:], (20, 20), order='F'))


#plot images and save figure
f, axes = plt.subplots(5, 5, figsize=(15, 10))
axes = axes.flatten()[:len(im_list)]
for img, ax in zip(im_list, axes.flatten()):
    img = img.squeeze()
    ax.imshow(img, 'gray')
plt.savefig('output/ps8-1-a')


#split the data between testing and training data
mnist_x_train, mnist_x_test, mnist_y_train, mnist_y_test = train_test_split(mnist_x, mnist_y, train_size=0.9, random_state=17)

#separate training data randomly through bagging (using test indicies of Kfold will give 900 elements randomly)
kf = KFold(n_splits=5, shuffle=True, random_state=17)
for i, (extra, train_index) in enumerate(kf.split(mnist_x_train)):
    x_train = mnist_x_train[train_index, :]
    y_train = mnist_y_train[train_index]

    #build dictionary for matlab files
    mdic = {"X_train": x_train, "y_train": y_train}
    filename = "input/mnist_fold_%d.mat" %(i + 1)
    sio.savemat(filename, mdic)

mdic = {"X_test": mnist_x_test, "y_test": mnist_y_test}
sio.savemat("input/mnist_test.mat", mdic)


#read in created .mat files to get consistent results
mnist_fold_1 = sio.loadmat('input/mnist_fold_1.mat')
mnist_fold_1_X = np.array(mnist_fold_1['X_train'])
mnist_fold_1_y = np.array(mnist_fold_1['y_train']).squeeze()
mnist_fold_2 = sio.loadmat('input/mnist_fold_2.mat')
mnist_fold_2_X = np.array(mnist_fold_2['X_train'])
mnist_fold_2_y = np.array(mnist_fold_2['y_train']).squeeze()
mnist_fold_3 = sio.loadmat('input/mnist_fold_3.mat')
mnist_fold_3_X = np.array(mnist_fold_3['X_train'])
mnist_fold_3_y = np.array(mnist_fold_3['y_train']).squeeze()
mnist_fold_4 = sio.loadmat('input/mnist_fold_4.mat')
mnist_fold_4_X = np.array(mnist_fold_4['X_train'])
mnist_fold_4_y = np.array(mnist_fold_4['y_train']).squeeze()
mnist_fold_5 = sio.loadmat('input/mnist_fold_5.mat')
mnist_fold_5_X = np.array(mnist_fold_5['X_train'])
mnist_fold_5_y = np.array(mnist_fold_5['y_train']).squeeze()
mnist_test = sio.loadmat('input/mnist_test.mat')
mnist_test_X = np.array(mnist_test['X_test'])
mnist_test_y = np.array(mnist_test['y_test']).squeeze()

#create a list to iterate through for accuracy calculations
mnist_fold_X = [mnist_fold_1_X, mnist_fold_2_X, mnist_fold_3_X, mnist_fold_4_X, mnist_fold_5_X, mnist_test_X]
mnist_fold_y = [mnist_fold_1_y, mnist_fold_2_y, mnist_fold_3_y, mnist_fold_4_y, mnist_fold_5_y, mnist_test_y]

#Train a One-vs-One SVM 10-class classifier
svm_clf = svm.SVC(kernel='poly', degree=3)
svm_clf.fit(mnist_fold_1_X, mnist_fold_1_y)
class_error(mnist_fold_X, mnist_fold_y, svm_clf, "SVM")

#Train a KNN (k=3) classifier
neigh_clf = KNeighborsClassifier(n_neighbors=3)
neigh_clf.fit(mnist_fold_2_X, mnist_fold_2_y)
class_error(mnist_fold_X, mnist_fold_y, neigh_clf, "KNN")

#Train a logistic regression classifier
logreg_clf = LogisticRegression()
logreg_clf.fit(mnist_fold_3_X, mnist_fold_3_y)
class_error(mnist_fold_X, mnist_fold_y, logreg_clf, "Logistic Regression")

#Train a decision tree classifier
dectree_clf = DecisionTreeClassifier()
dectree_clf.fit(mnist_fold_4_X, mnist_fold_4_y)
class_error(mnist_fold_X, mnist_fold_y, dectree_clf, "Decision Tree")

#Train a random forest
rdtree_clf = RandomForestClassifier()
rdtree_clf.fit(mnist_fold_5_X, mnist_fold_5_y)
class_error(mnist_fold_X, mnist_fold_y, rdtree_clf, "Random Forest")

#start the voting classifier by making predictions on the test set
svm_pred = svm_clf.predict(mnist_test_X)
neigh_pred = neigh_clf.predict(mnist_test_X)
logreg_pred = logreg_clf.predict(mnist_test_X)
dectree_pred = dectree_clf.predict(mnist_test_X)
rdtree_pred = rdtree_clf.predict(mnist_test_X)
vot_pred = np.empty(svm_pred.shape)

#get the class which occurs most often
for i in range(len(vot_pred)):
    vot_pred[i] = st.mode([svm_pred[i], neigh_pred[i], logreg_pred[i], dectree_pred[i], rdtree_pred[i]])[0]

#get the error of the testing set
print(f'Prediction error of Majority Voting on Testing Test: ',(1 - accuracy_score(vot_pred, mnist_test_y))*100,'%\n')
'''

#---------------------------------------------------------
#                   Problem 2
#---------------------------------------------------------

'''
#resize images
mpimg.thumbnail('input/HW8_F_images/im1.jpg', 'input/HW8_F_images/im1_thumbnail.jpg', scale=0.1)
mpimg.thumbnail('input/HW8_F_images/im2.jpg', 'input/HW8_F_images/im2_thumbnail.jpg', scale=0.5)
mpimg.thumbnail('input/HW8_F_images/im3.png', 'input/HW8_F_images/im3_thumbnail.png', scale=0.1)
'''

#load in images
im1 = mpimg.imread('input/HW8_F_images/im1_thumbnail.jpg')
im2 = mpimg.imread('input/HW8_F_images/im2_thumbnail.jpg')
im3 = mpimg.imread('input/HW8_F_images/im3_thumbnail.png')
im3[:,:,:-1]

#resize images as 2d arrays
im1.resize((im1.shape[0]*im1.shape[1], 3))
im2.resize((im2.shape[0]*im2.shape[1], 3))
im3.resize((im3.shape[0]*im3.shape[1], 3))

#create buckets for results and paramters
k_list = [3, 5, 7]
iters_list = [7, 13, 20]
r_list = [5, 15, 25]
im1_results = []
im2_results = []
im3_results = []

'''
#produce output images for im1
for k in k_list:
    for iters in iters_list:
        for r in r_list:
            #do kmeans on the image with new parameters
            membership, clusters, distance = kmeans_multiple(im1, k, iters, r)
            clusters = clusters.round()
            im1_out = np.zeros(im1.shape)

            #assign rgb values
            for pix in range(im1.shape[0]):
                im1_out[pix,:] = clusters[membership[pix]-1, :]

            #reshape image back to normal size and add to array
            im1_out = im1_out.reshape((78, 140, 3)).astype(int)
            im1_results.append(im1_out)

            #create image title
            title = f'output/im1/Image1_K{k}_iters{iters}_R{r}.png'

            print(title)

            #save image
            plt.imshow(im1_out)
            plt.savefig(title)

            
#repeat for im2
for k in k_list:
    for iters in iters_list:
        for r in r_list:
            #do kmeans on the image with new parameters
            membership, clusters, distance = kmeans_multiple(im2, k, iters, r)
            clusters = clusters.round()
            im2_out = np.zeros(im2.shape)

            #assign rgb values
            for pix in range(im2.shape[0]):
                im2_out[pix,:] = clusters[membership[pix]-1, :]

            #reshape image back to normal size and add to array
            im2_out = im2_out.reshape((100, 137, 3)).astype(int)
            im2_results.append(im2_out)

            #create image title
            title = f'output/im2/Image2_K{k}_iters{iters}_R{r}.png'

            print(title)

            #save image
            plt.imshow(im2_out)
            plt.savefig(title)
'''
            
#repeat for im3
for k in k_list:
    for iters in iters_list:
        for r in r_list:
            #do kmeans on the image with new parameters
            membership, clusters, distance = kmeans_multiple(im3, k, iters, r)
            clusters = clusters
            im3_out = np.zeros(im3.shape)

            #assign rgb values
            for pix in range(im3.shape[0]):
                im3_out[pix,:] = clusters[membership[pix]-1, :]

            #reshape image back to normal size and add to array
            im3_out = im3_out.reshape((75, 150, 3))
            im3_results.append(im3_out)

            #create image title
            title = f'output/im3/Image3_K{k}_iters{iters}_R{r}.png'

            print(title)

            #save image
            plt.imshow(im3_out)
            plt.savefig(title)