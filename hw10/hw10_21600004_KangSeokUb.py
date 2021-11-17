import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random as rd
import pandas as pd
import copy
import time


def euclidean_dist(a, b):
    return np.sqrt(np.sum((a - b)**2))

def kmeans(X, K):
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
            clustered_group = X[clusters==k]
            if len(clustered_group) > 0:
                centroids[k] = np.mean(clustered_group, axis=0)
    
        # temination status
        if(euclidean_dist(centroids, prev_centroids) == 0):
            break
    
    stds = np.zeros(K)

    std_of_stds = []
    for i in range(K):
        p = X[clusters == i]
        stds[i] = np.std(X[clusters == i])
        std_of_stds.append(stds[i])
    
    std_of_stds = np.mean(std_of_stds)
    
    for i in range(K):
        if(len(clusters == i) == 0):
            stds[i] = std_of_stds

    print("stds:", stds)
    return centroids, stds

class RBFNet(object):
    def __init__(self, K=2, lr=0.01, epochs=100):
        self.k = K
        self.lr = lr
        self.epochs = epochs
        
        self.w = np.random.randn(K)
        self.b = np.random.randn(1)
    
    def RBF(self, x, centorid, std):
        if std == 0.0:
            return 1
        else:
            return np.exp(-1 / (2 * std * std) * np.sum((x-centorid)**2))
        
    def fit(self, X, y):
        self.centroids, self.stds = kmeans(X, self.k)

        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                pi = np.array([self.RBF(X[i], centorid, std) for centorid, std, in zip(self.centroids, self.stds)])
                y_hat = self.w.T @ pi + self.b
                
                error = -(y[i] - y_hat)

                self.w = self.w - self.lr * pi * error
                self.b = self.b - self.lr * error

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            pi = np.array([self.RBF(X[i], centorid, std) for centorid, std, in zip(self.centroids, self.stds)])
            y_hat = self.w.T @ pi + self.b
            y_pred.append(y_hat)
            
        return np.array(y_pred)

def load_data(dataset):
    train1 = dataset + "_train1.txt"
    train2 = dataset + "_train2.txt"
    test = dataset +"_test.txt"
    
    train_dataset1 = pd.read_csv(train1, sep="\t",header=None)
    train_dataset2 = pd.read_csv(train2, sep="\t",header=None)
    
    test_dataset = pd.read_csv(test, sep="\t",header=None)
    
    return train_dataset1.to_numpy(), train_dataset2.to_numpy(), test_dataset.to_numpy()    
if __name__ == '__main__':
    ##################
    #      cis       #
    ##################
    train_cis_1, train_cis_2, test_cis = load_data("cis")
    
    test_cis_x_1, test_cis_y_1 = train_cis_1[:, :2], train_cis_1[:, 2:]
    test_cis_x_2, test_cis_y_2 = train_cis_2[:, :2], train_cis_2[:, 2:]
    test_cis_x, test_cis_y = test_cis[:, :2], test_cis[:, 2:]
    
    plt.figure(figsize=(10, 10))
    plt.scatter(test_cis_x[:, :1], test_cis_x[:, 1:2], c = test_cis_y, cmap = mcolors.ListedColormap(["black", "white"]))
    plt.title('cis_test Actual data')
    plt.show()
    
        
    ##################
    #     cis #1     #
    ##################
    rbfn_cis_1 = RBFNet(lr=1e-2, K=11, epochs=300)
    rbfn_cis_1.fit(test_cis_x_1, test_cis_y_1)

    y_pred = rbfn_cis_1.predict(test_cis_x)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    
    accuracy = sum(np.equal(test_cis_y, y_pred)) / len(test_cis_y)

    plt.figure(figsize=(10, 10))
    plt.scatter(test_cis_x[:, :1], test_cis_x[:, 1:2], c = y_pred, cmap = mcolors.ListedColormap(["black", "white"]))
    plt.text(0.3, 0.5, 'Accuracy:'+str(accuracy),  color='b')
    plt.title('cis_test#1')
    plt.show()
    
    
    ##################
    #     cis #2     #
    ##################
    rbfn_cis_2 = RBFNet(lr=1e-2, K=11, epochs=300)
    rbfn_cis_2.fit(test_cis_x_2, test_cis_y_2)

    y_pred = rbfn_cis_2.predict(test_cis_x)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    
    accuracy = sum(np.equal(test_cis_y, y_pred)) / len(test_cis_y)

    plt.figure(figsize=(10, 10))
    plt.scatter(test_cis_x[:, :1], test_cis_x[:, 1:2], c = y_pred, cmap = mcolors.ListedColormap(["black", "white"]))
    plt.text(0.3, 0.5, 'Accuracy:'+str(accuracy),  color='b')
    plt.title('cis_test#2')
    plt.show()
    
    
    ##################
    #       fa       #
    ##################
    train_fa_1, train_fa_2, test_fa = load_data("fa")
    
    train_fa_x_1, train_fa_y_1 = train_fa_1[:, :1], train_fa_1[:, 1:]
    train_fa_x_2, train_fa_y_2 = train_fa_2[:, :1], train_fa_2[:, 1:]
    test_fa_x, test_fa_y = test_fa[:, :1], test_fa[:, 1:]
    
    
    ##################
    #     fa #1      #
    ##################
    rbfn_fa = RBFNet(lr=1e-2, K=5, epochs=500)
    rbfn_fa.fit(train_fa_x_1, train_fa_y_1)
    
    y_pred = rbfn_fa.predict(test_fa_x)
    MSE = 1 / len(test_fa_y) * np.sum((test_fa_y - y_pred)**2)
    
    plt.plot(test_fa_x, test_fa_y, c='r', label="Aactual Data")
    plt.plot(test_fa_x, y_pred, c='b', label="Predicted Data")
    plt.title('fa test #1')
    plt.text(0.6, 0.1, 'MSE:%.5f'%MSE,  color='k')
    plt.show()
    
    ##################
    #     fa #2     #
    ##################
    rbfn_fa = RBFNet(lr=1e-2, K=5, epochs=500)
    rbfn_fa.fit(train_fa_x_2, train_fa_y_2)
    
    y_pred = rbfn_fa.predict(test_fa_x)
    MSE = 1 / len(test_fa_y) * np.sum((test_fa_y - y_pred)**2)
    plt.plot(test_fa_x, test_fa_y, c='r', label="Aactual Data")
    plt.plot(test_fa_x, y_pred, c='b', label="Predicted Data")
    plt.title('fa test #2')
    plt.text(0.6, 0.1, 'MSE:%.5f'%MSE,  color='k')
    plt.show()
    