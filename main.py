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
import _pickle as pickle
import os.path

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

#load data from given files and save it as tensors to speed up next executions
def load_data(trainDataFile,testDataFile):
    if not os.path.isfile('data_tensors.p'):
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

        data = [x_train,y_train,x_test,y_test]
        pickle.dump(data, open('data_tensors.p', 'wb'))
        return data
    else:
        return pickle.load(open('data_tensors.p', 'rb'))


'''
def oneHot(x, nbClass):
    xOneHot = np.zeros((len(x), nbClass))
    xOneHot[np.arange(len(x)), x] = 1
    return xOneHot
'''
'''
def evaluation(model,N,Xeval,Yeval):
    loss = 0
    acc = 0
    #Nbatch = 2048
    for k in range(N // Nbatch):
        
        indsBatch = range(k * Nbatch, (k+1) * Nbatch)
        X = Variable(Xeval[indsBatch, :], requires_grad=False)
        Y = Variable(Yeval[indsBatch, :], requires_grad=False)
        
        X,Y = X.cuda(), Y.cuda()
        
        Ytil = model(X)
        _, Y_not_onehot = Y.max(1)
        l, a = loss_accuracy(Ytil, Y_not_onehot, Y)
        loss += l
        acc += a
        #print acc
    return loss.cpu().data.numpy() / float(k+1), acc.cpu().data.numpy() / float(k+1)
'''
if __name__ == '__main__':
    trainDataFile = "mnist_rotation_new/mnist_all_rotation_normalized_float_train_valid.amat"
    testDataFile = "mnist_rotation_new/mnist_all_rotation_normalized_float_test.amat"

    print('loading data...',end='',flush=True)
    x_train,y_train,x_test,y_test = load_data(trainDataFile, testDataFile)
    
    #shuffle train and test data
    print('done',flush=True)
    print(y_train.shape)
    print(y_test.shape)
    print(x_train.shape)
    print(x_train.shape)

    #hyperparameters
    learning_rate = 0.05
    batch_size = 32
    train_size = x_train.shape[0]
    epochs_nb = 2
    cuda = True
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

    #book keeping
    nbBatchTotal = 0
    train_losses = []
    train_acc = []
    test_losses = []
    test_acc = []

    #training
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
            train_losses.append(l.data[0])
            optim.zero_grad()
            l.backward()
            optim.step()

        plt.plot(train_losses)
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