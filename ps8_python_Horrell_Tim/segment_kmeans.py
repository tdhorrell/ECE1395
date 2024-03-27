#segment_kemans.py

import numpy as np
from kmeans_multiple import kmeans_multiple

def segment_kmeans(im_in, K, iters, R):
    #initialization
    im_in_shape = np.shape(im_in)
    print(im_in_shape)
    im_in_resize = np.array(np.reshape(im_in, (im_in_shape[0]*im_in_shape[1], 3)))
    print(im_in_resize.shape)

    #do kmeans clustering
    membership, clusters, distance = kmeans_multiple(im_in, K, iters, R)
    clusters = clusters.round()

    #prepare image for output
    im_in_out = np.zeros(im_in_resize.shape)
    for pix in range(im_in_resize.shape[0]):
        im_in_out[pix,:] = clusters[membership[pix]-1, :]
    im_in_out = im_in_out.reshape((im_in_shape[0], im_in_shape[1], 3)).astype(int)

    return im_in_out