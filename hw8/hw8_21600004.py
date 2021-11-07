from pickle import TRUE
import six.moves.cPickle as pickle
import gzip
import os
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import pandas as pd
import copy
import torch
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

class MyKNN:
    def __init__(self):
        self.k = 1
    
    def fit(self, train_x, train_y, test_x):
        N_train, d_train = train_x.shape
        N_test, d_test = test_x.shape
        
        self.train_y = train_y
        
        print("N_train: {0}, d_train: {1}".format(N_train, d_train))
        print("N_test : {0}, d_test : {1}".format(N_test, d_test))
        
        # calculate distance 
        self.distance_Map = np.zeros((N_test, N_train))
        
        for i in range(N_test):
            for j in range(N_train):
                self.distance_Map[i][j] = np.sqrt(np.sum((train_x[i] - test_x[j])**2))
                
    def predict(self, test_y, K):
        predicted = []
        
        for idx, label in enumerate(test_y):
            near_distance = np.argsort(self.distance_Map[idx])
            vote = [0 for x in range(10)]
            
            for k in range(K):
                vote[self.train_y[near_distance[k]]] += 1
                
            predicted.append(np.argmax(vote))
            
        return predicted



# def knn(X_train, Y_train, X_test, K=3):
#     N_train, d_train = X_train.shape
#     N_test, d_test = X_test.shape
    
    
#     print("N_train: {0}, d_train: {1}".format(N_train, d_train))
#     print("N_test : {0}, d_test : {1}".format(N_test, d_test))
    
    
#     predicted = []
    
#     # calculate distance 
#     distance_Map = np.zeros((N_test, N_train))
    
#     for i in range(N_test):
#         for j in range(N_train):
#             distance_Map[i][j] = calc_distacne(X_test[i], X_train[j])
        
#         near_distance = np.argsort(distance_Map[i])
#         vote = [0 for x in range(10)]
        
#         for k in range(K):
#             vote[Y_train[near_distance[k]]] += 1
            
#         predicted.append(np.argmax(vote))
       
        
    
#     return predicted
        
            

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
    
    filtered_train_x = np.empty([0, 784])
    filtered_train_y = np.array([], dtype=int)
    
    
    for idx in range(10):
        x_mask = np.where(train_y == idx)
        x = train_x[x_mask][:1000]
        y = train_y[x_mask][:1000]
        
        filtered_train_x = np.append(filtered_train_x, x, axis = 0)
        filtered_train_y = np.append(filtered_train_y, y, axis = 0)
    
    print("train: {0} vs {1}".format(train_x.shape, filtered_train_x.shape))
    print("train: {0} vs {1}".format(train_y.shape, filtered_train_y.shape))
    
    print("Test: ", test_x.shape)
    
    # PCA
    mean_Train = test_x.mean(0)
    
    cov = np.cov(train_x.T)
    eig_val, eigVector = np.linalg.eig(cov)

    sorted_eigVector = eigVector[np.argsort(eig_val)[::-1]]
    sorted_eigVal = np.sort(eig_val)[::-1]
    
    list_test_y = test_y.tolist()
    
    K_list = [1, 5, 10]
    dim_list = [2, 7, 784]
    
    
    for dim in dim_list:
        knn = MyKNN()
        
        pca_dim_train = np.matmul(filtered_train_x, sorted_eigVector[:, :dim]).real
        pca_dim_test = np.matmul(test_x, sorted_eigVector[:, :dim]).real
        
        knn.fit(pca_dim_train, filtered_train_y, pca_dim_test)
        
        for k in K_list:
            predicted_y = knn.predict(test_y=test_y, K=k)
            
            file_name = "./predicted_y_dim" + str(dim) + "_k" + str(k) +".txt"
            f = open(file_name, "w")
            
            for val in predicted_y:
                f.write(str(val)+"\n")
            f.close()
            
            # test accuracy
            acc = 0
            for idx in range(len(list_test_y)):
                if(int(list_test_y[idx]) == int(predicted_y[idx])):
                    acc += 1
            
            print("Accuracy:: {0}".format(acc / len(list_test_y)))
            file_name = "./predicted_y_dim" + str(dim) + "_k" + str(k) +".txt"
            f = open(file_name, "w")
            
            for val in predicted_y:
                f.write(str(val)+"\n")
            f.close()

    
        
    
    
    