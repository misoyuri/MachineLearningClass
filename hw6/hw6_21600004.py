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

def visualization_scatter(X, Y, number_of_label, save_file_name):
    plt.figure(figsize=(13, 10))
    for label in range(number_of_label):
        mask = np.equal(Y, label)
        plt.scatter(X[mask, 0], X[mask, 1], label=str(label))
    plt.legend()
    plt.savefig("./images/" + save_file_name + ".png")
    plt.clf()

if __name__ == '__main__':
    train_set, val_set, test_set = load_data('mnist.pkl.gz')

    train_x, train_y = train_set
    
    # PCA
    mean_Train = train_x.mean(0)
    
    cov = np.cov(train_x.T)
    eig_val, eigVector = np.linalg.eig(cov)

    sorted_eigVector = eigVector[np.argsort(eig_val)[::-1]]
    
    pca_dim2 = np.matmul(train_x, sorted_eigVector[:, :2]).real
    
    visualization_scatter(pca_dim2, train_y, number_of_label=10, save_file_name="PCA")
    
    
    
    # LDA
    global_mean = train_x.mean(0).reshape(784, 1)
    N, d = train_x.shape

    class_data = []
    class_mean_data = []
    
    # make classified data list
    for label in range(10):
        class_data.append(train_x[train_y==label])
        class_mean_data.append(np.mean(class_data[label], axis=0).reshape(784, 1))
        
    sb = np.zeros((784, 784))


    # Calculate S of b
    for label in range(10):
        diff = class_mean_data[label] - global_mean
        sb += class_data[label].shape[0] * np.matmul(diff, diff.T)    
        
    # Calculate S of w
    sw = np.zeros((784, 784))
    for label in range(10):
        si = np.zeros((784, 784))
        for data_x in class_data[label]:
            diff = data_x.reshape(784, 1) - class_mean_data[label]
            si += np.matmul(diff, diff.T)
        sw += si
    
    print("class_mean_data::", class_mean_data[0].shape)
    print("global_mean::", global_mean.shape)
    print("sb::", sb.shape)
    print("sw::", sw.shape)
    
    invSw = np.linalg.pinv(sw)
    invSw_mul_Sb = np.matmul(invSw, sb)
    eig_val, eig_vec = np.linalg.eig(invSw_mul_Sb)
    sorted_eig_vec = eig_vec[:, np.argsort(eig_val)[::-1]]

    lda_dim2 = np.matmul(train_x, sorted_eig_vec[:, :2]).real
    visualization_scatter(lda_dim2, train_y, number_of_label=10, save_file_name="LDA")