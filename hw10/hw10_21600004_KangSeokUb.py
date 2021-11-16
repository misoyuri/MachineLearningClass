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
import time

class RBF:

    def __init__(self, X, y, tX, ty, num_of_classes,
                 k, std_from_clusters=True):
        self.X = X
        self.y = y

        self.tX = tX
        self.ty = ty

        self.number_of_classes = num_of_classes
        self.k = k
        self.std_from_clusters = std_from_clusters

    def convert_to_one_hot(self, x, num_of_classes):
        arr = np.zeros((len(x), num_of_classes))
        for i in range(len(x)):
            c = int(x[i])
            arr[i][c] = 1
        return arr

    def rbf(self, x, c, s):
        distance = euclidean_dist(x, c)
        return 1 / np.exp(-distance / s ** 2)

    def rbf_list(self, X, centroids, std_list):
        RBF_list = []
        for x in X:
            RBF_list.append([self.rbf(x, c, s) for (c, s) in zip(centroids, std_list)])
        return np.array(RBF_list)

    def fit(self):
        self.std_list, self.centroids = kmean(self.X, self.k)

        if not self.std_from_clusters:
            dMax = np.max([euclidean_dist(c1, c2) for c1 in self.centroids for c2 in self.centroids])
            self.std_list = np.repeat(dMax / np.sqrt(2 * self.k), self.k)

        RBF_X = self.rbf_list(self.X, self.centroids, self.std_list)

        self.w = np.linalg.pinv(RBF_X.T @ RBF_X) @ RBF_X.T @ self.convert_to_one_hot(self.y, self.number_of_classes)

        RBF_list_tst = self.rbf_list(self.tX, self.centroids, self.std_list)

        self.pred_ty = RBF_list_tst @ self.w
        print(self.pred_ty[-50:])

        self.pred_ty = np.array([np.argmax(x) for x in self.pred_ty])

        # diff = self.pred_ty - self.ty
        # print('Accuracy: ', len(np.where(diff == 0)[0]) / len(diff))

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

def load_data(dataset):
    train1 = dataset + "_train1.txt"
    train2 = dataset + "_train2.txt"
    test = dataset +"_test.txt"
    
    train_dataset1 = pd.read_csv(train1, sep="\t",header=None)
    train_dataset2 = pd.read_csv(train2, sep="\t",header=None)
    train_dataset = pd.concat([train_dataset1, train_dataset2])
    
    test_dataset = pd.read_csv(test, sep="\t",header=None)
    
    return train_dataset, test_dataset    
if __name__ == '__main__':
    train_cis, test_cis = load_data("cis")
    train_fa, test_fa = load_data("fa")

    train_cis = train_cis.to_numpy()
    test_cis = test_cis.to_numpy()
    train_fa = train_fa.to_numpy()
    test_fa = test_fa.to_numpy()
    
    train_cis_x, train_cis_y = train_cis[:, :2], train_cis[:, 2:]
    test_cis_x, test_cis_y = test_cis[:, :2], test_cis[:, 2:]
    
    print(train_cis_x[:5, :])
    print("train_cis_x: ", train_cis_x.shape)
    print("train_cis_y : ", train_cis_y.shape)
    print("*"*20)
    print("test_cis_x: ", test_cis_x.shape)
    print("test_cis_y: ", test_cis_y.shape)
    print("*"*20)
    print("test_cis_x: ", test_cis_x.shape)
    print("test_cis_y: ", test_cis_y.shape)
    print("*"*20)
    print("train_fa: ", train_fa.shape)
    print("test_fa : ", test_fa.shape)
    
    # train_cis_x = train_cis_x.to_numpy()
    plt.plot(train_cis_x[:, :1], train_cis_x[:, 1:2], marker='o')
    plt.show()
    
    # RBF_CLASSIFIER = RBF(train_cis_x, train_cis_y, test_cis_x, test_cis_y, num_of_classes=2, k=2, std_from_clusters=False)
    # RBF_CLASSIFIER.fit()