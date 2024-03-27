#kmeans_single.py

import numpy as np
from scipy.spatial.distance import cdist


def compute_membership(X, clusters):
    membership = []
    for val in X:
        distance = cdist(clusters, np.reshape(val, (1, X.shape[1])))
        membership.append(np.argmin(distance) + 1)

    return membership

def compute_means(X, clusters, membership):
    membership_sum = np.zeros(clusters.shape)
    membership_count = np.zeros(clusters.shape[0])

    for i in range(X.shape[0]):
        membership_sum[membership[i]-1, :] += X[i, :]
        membership_count[membership[i] - 1] += 1

    for i in range(clusters.shape[0]):
        if(membership_count[i] != 0):
            clusters[i, :] = membership_sum[i] / membership_count[i]

    return clusters

def compute_distance(X, clusters, membership):
    total_dist = 0
    for i, val in enumerate(X):
        distance = cdist(np.reshape(clusters[membership[i]-1], (1,X.shape[1])), np.reshape(val, (1, X.shape[1])))
        total_dist += distance

    return total_dist

def kmeans_single(X, K, iters):
    #initialize variables
    X = np.array(X)
    range_min = []
    range_max = []
    
    #get K cluster centers
    clusters = []

    #find range of random values to randomize between
    for i in range(X.shape[1]):
        range_min.append(np.min(X[:,i]))
        range_max.append(np.max(X[:,i]))

    #use rng to generate K clusters across all features (K x n)
    for i in range(K):
        temp_cluster = []
        for j in range(X.shape[1]):
            temp_cluster.append(np.random.uniform(range_min[j], range_max[j], 1))
        if(i == 0):
            clusters = np.transpose(temp_cluster)
        else:
            clusters = np.vstack((clusters, np.transpose(temp_cluster)))

    for iter in range(iters):
        #update membership
        membership = compute_membership(X, clusters)
        #recalculate clusters
        clusters = compute_means(X, clusters, membership)
        #calculate SSD
        distance = compute_distance(X, clusters, membership)

    return membership, clusters, distance


