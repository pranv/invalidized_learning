import numpy as np
import os, sys, cPickle

from invalidizer.invalidizer import *

def load_batch(fpath, label_key='labels'):
    f = open(fpath, 'rb')
    d = cPickle.load(f)
    f.close()
    data = d["data"]
    labels = d[label_key]
    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


def load_data(path):
    dirname = "cifar-10-batches-py"
    
    nb_test_samples = 10000
    nb_train_samples = 50000
    
    X_train = np.zeros((nb_train_samples, 3, 32, 32), dtype="uint8")
    y_train = np.zeros((nb_train_samples,), dtype="uint8")
    
    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        X_train[(i-1)*10000:i*10000, :, :, :] = data
        y_train[(i-1)*10000:i*10000] = labels

    fpath = os.path.join(path, 'test_batch')
    X_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    return (X_train, y_train), (X_test, y_test)


path = '/Users/pran/Desktop/Deep Learning/Datasets/cifar-10-batches-py'

(X_train, y_train), (X_test, y_test) = load_data(path)

test = X_train[0172].swapaxes(0,2)

import matplotlib.pyplot as plt
plt.ion()

plt.imshow(test)
raw_input()

inval = invalidizer(test, 4, 8)

plt.imshow(inval)
raw_input()