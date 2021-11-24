from numpy.core.numeric import Inf
import six.moves.cPickle as pickle
import gzip
import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import random
from sklearn import metrics
import time

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
    
    return train_set, valid_set, test_set


def norm(X):
    return X - np.mean(X, axis=0)

def one_hot(y):
    one_hot = []
    oh = np.zeros(10, dtype=int)
    for i in y:
        oh[i] = 1
        one_hot.append(oh)
        oh = np.zeros(10, dtype=int)
    return np.asarray(one_hot)

def initp(x_dim, h, C):
    
    ##Initialize each array according to the corresponding dimension
    ## W@ are inistialized randomly, and b@ can be initialized as zero vectors (why can't W@ be zero?)

    W_xh = np.random.randn(x_dim, h) #complete the  ...
    b_h = np.zeros((1, h))
    Wo = np.random.randn(h, C) 
    bo = np.zeros((1, C))
    
    ##Save the parameter values as a dictionary. 
    ##This is our 'memory' where we store our parameters, and later we update them here.
    parameters = {"W_xh": W_xh,
                  "b_h": b_h,
                  "Wo": Wo,
                  "bo": bo}
    
    return parameters

def softmax(x):
    if(np.max(x) == 0):
        print("x is zero shibal\n",x)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dSigmoid(x):
    return Sigmoid(x) * (1 - Sigmoid(x))

def leaky_ReLU(x):
    return np.maximum(0.01 * x, x)

def der_leaky_ReLU(x):
    dx = np.ones_like(x)
    dx[x < 0] = 0.01
    
    return dx
    
    
def forward_prop(X, parameters):
    
    ##Let's load our current parameters
    W_xh = parameters["W_xh"]
    b_h = parameters["b_h"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    
    ##This is basically the MLP:
    ##hidden layer:
    Z1 = np.dot(W_xh.T, X) + b_h.T
    A1 = Sigmoid(Z1) ##replace your_act with an activation of your liking like ReLU or tanh
    ##output layer:
    Zo = np.dot(Wo.T, A1) + bo.T #complete the  ...
    Ao = softmax(Zo) #complete the  ...

    ##we make a cache, that is a 'memory' so that later we can go back in our reasoning when we do back propagation
    cache = {"Z1": Z1,
             "A1": A1,
             "Zo": Zo,
             "Ao": Ao}
    
    ##the output is our probabilities. 
    return Ao, cache

def nll_cost(Ao, Y, parameters):
    Ao[Ao == 0] = 0.000000000000000000000001
    n = Y.shape[0]

    logprobs = np.multiply(np.log(Ao), Y.T)
    cost =  (-1./n)*np.sum(logprobs)
    cost = np.squeeze(cost)     

    return cost

def back_prop(parameters, cache, X, Y):
    
    ##Amount of examples
    n = X.shape[0]

    ##Load current parameter weights
    W_xh = parameters["W_xh"]
    Wo = parameters["Wo"]
    
    ##Load the activation information of each layer
    A1 = cache["A1"]
    Ao = cache["Ao"]
    
    ##Let's compute the derivatives! Note we are going backwards, from the output layer to the hidden layer
    dZo= Ao - Y.T

    #dim of dWo should be (10, 200)
    dWo = (1./n)*np.dot(dZo, A1.T) #complete the  ...

    #dim of dbo should be (10, 1)
    dbo = (1./n)*np.sum(dZo, axis=1, keepdims=True) #complete the  ...

    # #dim of dZ1 should be (200, 50000)
    dZ1 = np.dot(dWo.T, dZo) * dSigmoid(A1)  #complete the information wrt the activation that you chose

    #dim of dW_xh should be (200, 784)
    dW_xh = (1./n)*np.dot(dZ1, X.reshape(50000, 784)) #complete the  ...
    
    #dim of db_h should be (200, 1)
    db_h = (1./n)*np.sum(dW_xh, axis=1, keepdims=True) #complete the  ...

    ##Save the gradients in a dictionary to update the parameters
    grads = {"dW_xh": dW_xh,
             "db_h": db_h,
             "dWo": dWo,
             "dbo": dbo}

    # print("X::", X.shape)
    # print("W_xh::", W_xh.shape)
    # print("Wo::", Wo.shape)
    # print("A1::", A1.shape)
    # print("Ao::", Ao.shape)
    # print("dZo::", dZo.shape)
    # print("dWo::", dWo.shape)
    # print("dbo::", dbo.shape)
    # print("dZ1::", dZ1.shape)
    # print("dW_xh::", dW_xh.shape)
    # print("db_h::", db_h.shape)


    return grads

def update(parameters, grads, lr):
    
    #Load your current parameters from the forward prop step
    W_xh = parameters["W_xh"]
    b_h = parameters["b_h"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    
    #Load the derivatives of your parameters from the backward step
    dW_xh = grads["dW_xh"]
    db_h = grads["db_h"]
    dWo = grads["dWo"]
    dbo = grads["dbo"]

    #Update your parameters using the gradient descent algorithm
    W_xh = W_xh - lr * dW_xh.T
    b_h = b_h - lr * db_h.T
    Wo = Wo - lr * dWo.T  # ... #complete the  ...
    bo = bo - lr * dbo.T  # ... #complete the  ...

    #Store your new parameters
    parameters = {"W_xh": W_xh,
                  "b_h": b_h,
                  "Wo": Wo,
                  "bo": bo}

    return parameters

def predict(prediction, labels):
    #get the label with the maximum probability in our estimation
    predictions = np.argmax(prediction, axis=0) 
    #check our estimation wrt the true label
    new = np.array([labels[i] == predictions[i] for i,_ in enumerate(labels)], dtype=np.bool)
    #compute the accuracy
    acc = np.count_nonzero(new*1)/labels.shape[0]
    
    return acc * 100

def MLP_model_train(X, train_y, H, O, D, lr, num_epochs, print_cost):
    
    ##initialize the parameters 
    parameters = initp(D, H, O)

    ##Load your initialized values
    W_xh = parameters["W_xh"]
    b_h = parameters["b_h"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]

    cost_per_epoch = []
    acc_per_epoch = []
    #Train!
    for i in range(0, num_epochs):
        #forward prop
        Ao, cache = forward_prop(X=X, parameters=parameters) #complete the  ...

        #compute the cost and save it
        cost = nll_cost(Ao=Ao, Y=one_hot(train_y), parameters=parameters) #complete the  ...
        cost_per_epoch.append(cost)

        #compute the ccuracy just for reference
        acc = predict(Ao, train_y) #complete the  ...
        acc_per_epoch.append(acc)
        #backpropagate!
        grads = back_prop(parameters=parameters, cache=cache, X=X, Y=one_hot(train_y)) #complete the  ...

        #update!
        parameters = update(parameters=parameters, grads=grads, lr=lr) #complete the  ...

        ##print the results every 10 epochs
        if print_cost and i % 10 == 0:
            print ("Cost after iteration %i: %f with accuracy of %f" %(i, cost, acc))
            
    print('Training ended.')
    return parameters, cost_per_epoch, acc_per_epoch

def test(parameters, test_im, test_labels):
    ##Forward prop your new images using your trained parameters
    a_test, cache = forward_prop(test_im, parameters)
    ##Check the accuracy
    acc = predict(a_test, test_labels)
    ##compute the cost function of the testing
    cost = nll_cost(a_test, one_hot(test_labels), parameters)
    return "Cost for test dataset: %f with accuracy of %f" %(cost, acc)

if __name__ == '__main__':
    

    
    train_set, val_set, test_set = load_data('./mnist.pkl.gz')

    ## I will not use a validation set, but you can if you want to :)
    ## train_x and test_x contain the images for training and testing. 
    train_x, train_y = train_set
    test_x, test_y = test_set
    print('The amount of images in your training set is: ', train_x.shape[0])
    print('The amount of images in your testing set is: ', test_x.shape[0])

    train_scaled = norm(train_x)
    test_scaled = norm(test_x)

    Y = one_hot(train_y)
    
    x_dim = 28 * 28 
    h = 200 
    C = 10 
    lr = 0.01
    num_epochs = 30000
    
    parameters = initp(x_dim, h, C) ## <- check the dimensions are correct!
    ao , cache = forward_prop(train_scaled.T, parameters)
    print("nll cost::", nll_cost(ao, one_hot(train_y), parameters))
    
    grads = back_prop(parameters, cache, train_scaled, one_hot(train_y))
    
    start_time = time.time()
    parameters, cost_per_epoch, acc_per_epoch = MLP_model_train(train_scaled.T, train_y, h, C, x_dim, lr, num_epochs, print_cost=True)
    print("---train {}s seconds---".format(time.time()-start_time))
     
    print(test(parameters, test_scaled.T, test_y))
    
    plt.plot(cost_per_epoch, c='r', label="Cost")
    plt.legend()
    plt.title('cost per epoch')
    plt.savefig("./cost_train.png")
    plt.clf()
    
    plt.plot(acc_per_epoch, c='b', label="Accuracy")
    plt.legend()
    plt.title('Accuracy per epoch')
    plt.savefig("./acc_train.png")
    plt.clf()
    
    plt.plot(cost_per_epoch, c='r', label="Cost")
    plt.plot(acc_per_epoch, c='b', label="Accuracy")
    plt.legend()
    plt.title('Accuracy and Cost per epoch')
    plt.savefig("./total.png")
    plt.clf()