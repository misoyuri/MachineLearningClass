from numpy.linalg.linalg import eig
import six.moves.cPickle as pickle
import gzip
import os
import numpy as np
import matplotlib.pyplot as plt
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


if __name__ == '__main__':
    train_set, val_set, test_set = load_data('mnist.pkl.gz')

    train_x, train_y = train_set
    val_x, val_y = val_set
    test_x, test_y = test_set
    
    print(train_x.shape)
    print(train_y.shape)
    
    meanTrain = np.mean(train_x, axis=0)
    varTrain = np.var(train_x, axis=0)
    
    mean_img = meanTrain.reshape((28,28))
    var_img = varTrain.reshape((28,28))
    
    plt.figure("mean")
    plt.imshow(mean_img, cmap='gray')
    plt.figure("var")
    plt.imshow(var_img, cmap='gray')
    
    cov = np.cov(train_x.T)
    eigValue, eigVector = np.linalg.eig(cov)
    
    
    print("cov::", cov.shape)
    print("eigeVaule::", eigValue.shape)
    print("eigeVector::", eigVector.shape)
    
    for i in range(10):
        vec_img = eigVector[i].reshape((28,28))*255.9
        vec_img = Image.fromarray(vec_img.astype(np.uint8))

        plt.figure("eigen vector image"+str(i))
        plt.imshow(vec_img, cmap='gray')
        
    plt.figure("eigen value 1 to 100")
    plt.plot(eigValue[:100])
    plt.show()
    
    # print(cov)
    # print(cov.shape)
    # # for eigendecomposition 
    # check http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html 
