from numpy.core.defchararray import equal
from numpy.linalg.linalg import eig
import six.moves.cPickle as pickle
import gzip
import os
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from numpy import linalg as LA

#import scipy.misc
from PIL import Image

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    
    copied from http://deeplearning.net/ and revised by hchoi
    '''

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    return train_set, valid_set, test_set

def euclidean_dist(a, b):
    return np.sqrt(np.sum(a - b)**2)

def kmean(x, k, max_iter=1000, threshold=0.001):
    N, d = x.shape
    centroid = np.zeros((k, d))

    sse = 0

    for idx in range(k):
        random_idx = rd.randrange(N)
        centroid[idx] = x[random_idx]

    for idx in range(max_iter):
        cluster_group = []
        for n in range(N):
            dist_centroid = []

            for k_idx in range(k):
                dist_centroid.append(euclidean_dist(x[n], centroid[k_idx]))
            cluster_group.append(np.argmin(dist_centroid)) 
        
        centroid = np.zeros((k, d))
        
        sum_ = []
        cnt_ = []
        
        for k_idx in range(k):
            sum_.append(np.zeros(d))
            cnt_.append(0)
        
        for n in range(N):
            sum_[cluster_group[n]] += x[cluster_group[n]]
            cnt_[cluster_group[n]] += 1
        
        for k_idx in range(k):
            centroid[k_idx] = sum_[k_idx] / cnt_[k_idx]
        
        prev_sse = sse
        sse = 0
        
        for n in range(N):
            sse += euclidean_dist(x[n], centroid[cluster_group[n]])**2
        
        print("sse::", sse)
        if prev_sse - sse < threshold :
            break



    return cluster_group, centroid, k

def visualization(X, clusters, centroid, K):    # visualization
    for k in range(K):
        mask = np.equal(clusters, k)
  
    plt.scatter(X[mask, 0], X[mask, 1])
    
    plt.scatter(centroid[:, 0], centroid[:, 1], marker = "x", s = 100)
    plt.show()

if __name__ == '__main__':
    train_set, val_set, test_set = load_data('mnist.pkl.gz')

    train_x, train_y = train_set
    val_x, val_y = val_set
    test_x, test_y = test_set
    
    train_x_mask = np.where((train_y == 3) | (train_y == 9))
    train_x = train_x[train_x_mask]
    
    cluster, centroid, k = kmean(train_x, 2)
    print(cluster)
    meanTrain = np.mean(train_x, axis=0)
    varTrain = np.var(train_x, axis=0)
    
    mean_img = meanTrain.reshape((28,28))
    var_img = varTrain.reshape((28,28))
    
    cov = np.cov(train_x.T)
    eigValue, eigVector = np.linalg.eig(cov)
    
    print("X : ", train_x.shape)
    print("ev: ", eigVector[:, 3].shape)
    
    visualization(train_x, cluster, centroid, k)
