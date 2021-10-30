import six.moves.cPickle as pickle
import gzip
import os
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import pandas as pd
import copy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
    test_x, test_y = test_set
    
    # PCA
    # mean_Train = train_x.mean(0)
    
    # cov = np.cov(train_x.T)
    # eig_val, eigVector = np.linalg.eig(cov)

    # sorted_eigVector = eigVector[np.argsort(eig_val)[::-1]]
    
    # pca_dim2 = np.matmul(train_x, sorted_eigVector[:, :2]).real
    
    # visualization_scatter(pca_dim2, train_y, number_of_label=10, save_file_name="PCA")
    print("TSNE Start")
    tsne_dim2 = TSNE(n_components= 2,random_state = 0).fit_transform(test_x)
    print("TSNE End")
    visualization_scatter(tsne_dim2, test_y, number_of_label=10, save_file_name="TSNE")
    