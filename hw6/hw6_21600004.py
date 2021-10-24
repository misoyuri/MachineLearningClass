import six.moves.cPickle as pickle
import gzip
import os
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import pandas as pd
import copy
from sklearn.decomposition import PCA
from numpy.linalg.linalg import eig, eigvals

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
    return np.sqrt(np.sum((a - b)**2))

def kmean(X, K):
    N, d = X.shape
    centroids = np.zeros((K, d))
    
    # init centroids
    for k in range(K):
        centroids[k] = X[rd.randrange(N)]
    
    # init clusters
    clusters = np.zeros(N)
    
    # repeat until centroids does not translate
    while(True):
        # assignment step
        for n in range(N):
            distance = []
            for k in range(K):
                distance.append(euclidean_dist(centroids[k], X[n]))
                
            clusters[n] = np.argmin(distance)
            
        # update step
        prev_centroids = copy.deepcopy(centroids)

        for k in range(K):
            clustered_group = []
            for n in range(N):
                if clusters[n] == k:
                    clustered_group.append(X[n])
            centroids[k] = np.mean(clustered_group, axis=0)
    
        # temination status
        if(euclidean_dist(centroids, prev_centroids) == 0):
            break
        
    return clusters, centroids

def print_cluster_info(X, Y, clusters, dim, K, centroid):
    members = []
    for k in range(K):
        members.append([])
        
    for n in range(len(X)):
        members[int(clusters[n])].append(X[n])
    
    print("*"*45)
    print("When Dimenstion is {0} and K of K-mean is {1}".format(dim, K))
    for idx, member in enumerate(members):
        print("# of {0}th cluster is {1}".format(idx, len(member)))
    
    if(K == 2):
        clustered = [[0, 0], [0, 0]]
        for n in range(len(X)):
            if(int(clusters[n]) == 0 and int(Y[n]) == 3):
                clustered[0][0] += 1
            elif(int(clusters[n]) == 0 and int(Y[n]) == 9):
                clustered[0][1] += 1
            elif(int(clusters[n]) == 1 and int(Y[n]) == 3):
                clustered[1][0] += 1
            elif(int(clusters[n]) == 1 and int(Y[n]) == 9):
                clustered[1][1] += 1

        print(clustered)
        print("*"*45)
    
    for k in range(K):
        fig = plt.figure()
        if dim == 784:
            subtitle = "Raw Image Clustering Result: K=" + str(K) + " - " + str(k) +"th clustered groupd"
            title = "RawImage_K" + str(K)+"_"+str(k)+"th"
        else:
            subtitle = "dim " + str(dim)+ " Image Clustering Result: K=" + str(K) + " - " + str(k) +"th clustered groupd"
            title = "dim"+str(dim)+"_K"+str(K)+"_"+str(k)+"th"
        for i in range(100):
            plottable_image = np.reshape(members[k][i], (28, 28))
            ax = fig.add_subplot(10, 10, i+1)
            ax.axis('off')
            ax.imshow(plottable_image, cmap='gray_r')
        plt.suptitle(subtitle)
        plt.savefig("./images/" + title + ".png")
        plt.clf()
        
# visualization only dim is 2
def visualization_scatter(X, clusters, centroid, K):
    for k in range(K):
        mask = np.equal(clusters, k)
        plt.scatter(X[mask, 0], X[mask, 1])
    
    plt.scatter(centroid[:, 0], centroid[:, 1], marker = "x", s = 100)
    plt.imshow()
    # plt.savefig("./images/" + "dimenstion_2_clustered_image_"+str(K) + ".png")
    # plt.clf()

if __name__ == '__main__':
    train_set, val_set, test_set = load_data('mnist.pkl.gz')

    train_x, train_y = train_set
    
    
    # PCA
    # mean_Train = train_x.mean(0)
    
    # cov = np.cov(train_x.T)
    # _, eigVector = np.linalg.eig(cov)

    # dim2_data = np.matmul(train_x, eigVector[:, :2]).real
    
    # K = 10
    # clusters, centroids = kmean(dim2_data, K)
    # visualization_scatter(dim2_data, clusters, centroids, K)
    
    # LDA
    df_train_x = pd.DataFrame(train_x)
    global_mean_train = train_x.mean(0)
    
    d = train_x[0].shape
    
    
    train_classified = []
    
    for idx in range(10):
        train_classified.append([])
    
    for x, y in zip(train_x, train_y):
        train_classified[int(y)].append(x)
    
    train_mean_classified = []
    
    for label in train_classified:
        np_label = np.array(label)
        train_mean_classified.append(np_label.mean(0))    
        
    sw = np.matrix((d, d))
    
    for label, classified_data in enumerate(train_classified):
        si = np.zeros((d, d))
        for x in classified_data:
            x_mi = np.matrix(x - train_mean_classified[label])
            si += x_mi @ x_mi.T
        sw += si
    print(sw.shape)