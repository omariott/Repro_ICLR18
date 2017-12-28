# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 11:00:43 2017

@author: rportelas
"""

import numpy as np
import matplotlib.pyplot as plt
import torch as t
from torch.autograd import Function

def read_data(fname):
    f = open(fname)
    f.readline()  # skip the header
    data = np.loadtxt(f)
    x = data[:, :-1]
    y = data[:, -1].astype(int)
    return x,y

def load_mnist(filename):
    with open(filename,"r") as f:
        f.readline()
        data =[ [float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp = np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)
    
def computeAccuracy(pred, y):
    _, predInd = t.max(pred, 1)
    _, trueInd = t.max(y, 1)
    correctPredictions = t.sum(predInd == trueInd)
    return correctPredictions / float(len(y))


def oneHot(x, nbClass):
    xOneHot = np.zeros((len(x), nbClass))
    xOneHot[np.arange(len(x)), x] = 1
    return xOneHot
  


if __name__ == '__main__':      
    print('loading data...',end='')
    trainDataFile = "mnist_rotation_new/mnist_all_rotation_normalized_float_train_valid.amat"
    testDataFile = "mnist_rotation_new/mnist_all_rotation_normalized_float_test.amat"

    xTrain, yTrain = read_data(trainDataFile)
    xTest, yTest = read_data(testDataFile)

    nb_class = len(np.unique(yTest))
    #one_hot encoding of labels
    yTrain = t.FloatTensor(oneHot(yTrain, nb_class))
    yTest = t.FloatTensor(oneHot(yTest, nb_class))

    xTrain = t.FloatTensor(xTrain)
    xTest = t.FloatTensor(xTest)
    print('done')
    print(yTrain.shape)
    print(yTest.shape)
    print(xTrain.shape)
    print(yTrain.shape)
    '''
    #training part
    nbBatchTotal = 0
    for e in xrange(nbEpochs):
        for nbBatch in range(nbBatchTrain):
        nbBatchTotal += nbBatch
    '''
    '''   
    plt.plot([j*batch_size for j in xrange(nbBatchTotal + nbEpochs)], lossData)
    plt.ylabel('evolution of Loss during training')
    plt.xlabel('training iteration number')
    plt.savefig('plots/lossMNIST_rot_im_lr0_05__4epochs.png')
    plt.show()
    plt.plot([j*batch_size for j in xrange(nbBatchTotal + nbEpochs)], accuracy)
    plt.plot([j*graphic_step for j in range(1, len(testAccuracy)+1)],testAccuracy)
    plt.xlabel('training iteration number')
    plt.legend(['train accuracy', 'test accuracy'], loc='lower right')
    plt.savefig('plots/accuracyMNIST_rot_im_lr0_05__4epochs.png')
    plt.ylabel('evolution of accuracy during training')
    plt.show()
    print 'max accuracy on test set: ' + str(max(testAccuracy))
    '''