#kmeans_multiple.py

import numpy as np
from kmeans_single import kmeans_single

def kmeans_multiple(X, K, iters, R):
    membership = []
    clusters = []
    distance = 0
    for r in range(R):
        #call single function
        temp_membership, temp_clusters, temp_distance = kmeans_single(X, K, iters)

        #take minimum distance
        if(r == 0):
            membership = temp_membership
            clusters = temp_clusters
            distance = temp_distance
        elif(temp_distance < distance):
            membership = temp_membership
            clusters = temp_clusters
            distance = temp_distance

    return membership, clusters, distance