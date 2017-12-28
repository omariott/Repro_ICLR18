# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 11:00:43 2017

@author: rportelas
"""

import numpy as np
import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn
from torch.autograd import Function as F
from torch.autograd import Variable

class baseDNN(nn.Module):

    def __init__(self,dim_x,dim_y,h):
        nn.Module.__init__(self)
        
        self.f1 = nn.Linear(dim_x,dim_h)
        self.f1.weight.data.uniform_(-0.1,0.1)
        
        self.f2 = nn.Linear(dim_h,dim_y)
        self.f2.weight.data.uniform_(-0.1,0.1)
        
    def forward(self,x):
        h_out = F.relu(self.f1(x))
        out = self.f2(h_out)
        return F.log_softmax(out,dim=0)


def read_data(fname):
    f = open(fname)
    f.readline()  # skip the header
    data = np.loadtxt(f)
    x = data[:, :-1]
    y = data[:, -1].astype(int)
    return x,y

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
    print('loading data...',end='',flush=True)
    trainDataFile = "mnist_rotation_new/mnist_all_rotation_normalized_float_train_valid.amat"
    testDataFile = "mnist_rotation_new/mnist_all_rotation_normalized_float_test.amat"

    x_train, y_train = read_data(trainDataFile)
    x_test, y_test = read_data(testDataFile)

    nb_class = len(np.unique(y_test))
    #one_hot encoding of labels
    #y_train = t.FloatTensor(oneHot(y_train, nb_class))
    #y_test = t.FloatTensor(oneHot(y_test, nb_class))
    x_train = t.FloatTensor(x_train)
    y_train = t.LongTensor(y_train)

    x_test = t.FloatTensor(x_test)
    y_test = t.LongTensor(y_test)
    

    #shuffle train and test data
    print('done',flush=True)
    print(y_train.shape)
    print(y_test.shape)
    print(x_train.shape)
    print(y_train.shape)

    #hyperparameters
    learning_rate = 0.05
    batch_size = 32
    train_size = x_train.shape[0]
    epochs_nb = 1
    cuda = False
    #graph_step = batch_size * 100

    #WARNING - Task specific
    input_dim = 784
    output_dim = 10

    #Baseline DNN settings, according to paper
    model = model = nn.Sequential(
          nn.Linear(input_dim,312),
          nn.ReLU(),
          nn.Linear(312,128),
          nn.ReLU(),
          nn.Linear(128,10),
        )
    loss = nn.CrossEntropyLoss()

    if cuda:
        model = model.cuda()
        loss = loss.cuda()

    optim = t.optim.SGD(model.parameters(), lr=learning_rate)

    #training part
    nbBatchTotal = 0
    losses = []
    for e in range(epochs_nb):
        for i in range(train_size // batch_size):
            #load batch
            indsBatch = range(i * batch_size, (i+1) * batch_size)
            x = Variable(x_train[indsBatch, :], requires_grad=False)
            y = Variable(y_train[indsBatch], requires_grad=False) 
            if cuda: x,y = x.cuda(), y.cuda()

            #forward
            y_til = model(x)
            #loss and backward
            l = loss(y_til,y)
            losses.append(l.data[0])
            optim.zero_grad()
            l.backward()
            optim.step()

            #if (i % graph_step) == 0:
            plt.plot(losses)
            plt.xlabel('nb mini-batch updates')
            plt.ylabel('loss')
            plt.show(block=False)
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