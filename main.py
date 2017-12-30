# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 11:00:43 2017

@author: rportelas
"""

import numpy as np
import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import _pickle as pickle
import os.path

from DEN import DEN

class baseDNN(nn.Module):

    def __init__(self,dim_x,dim_y,dim_h=100):
        nn.Module.__init__(self)

        self.f1 = nn.Linear(dim_x,dim_h)
        self.f1.weight.data.uniform_(-0.1,0.1)

        self.f2 = nn.Linear(dim_h,dim_y)
        self.f2.weight.data.uniform_(-0.1,0.1)

    def forward(self,x):
        h_out = F.relu(self.f1(x))
        out = self.f2(h_out)
        return F.log_softmax(out,dim=0)

    def batch_pass(self, x_train, y_train, loss, optim, batch_size=32, cuda=False):
        for i in range(train_size // batch_size):
            #load batch
            indsBatch = range(i * batch_size, (i+1) * batch_size)
            x = Variable(x_train[indsBatch, :], requires_grad=False)
            y = Variable(y_train[indsBatch], requires_grad=False)
            if cuda: x,y = x.cuda(), y.cuda()

            #forward
            y_til = self.forward(x)
            #loss and backward
            l = loss(y_til,y)
            optim.zero_grad()
            l.backward()
            optim.step()

def read_data(fname):
    f = open(fname)
    f.readline()  # skip the header
    data = np.loadtxt(f)
    x = data[:, :-1]
    y = data[:, -1].astype(int)
    return x,y

def computeAccuracy(pred, y):
    _, predInd = t.max(pred, 1)
    correctPredictions = t.sum(predInd == y)
    return (correctPredictions.data[0] / y.shape[0])

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

def evaluation(model, x_eval,y_eval):
    acc = 0
    l = 0
    nb_iter = x_eval.shape[0]//2048
    for k in range(nb_iter):
        #load batch
        indsBatch = range(k * batch_size, (k+1) * batch_size)
        x = Variable(x_eval[indsBatch, :], requires_grad=False)
        y = Variable(y_eval[indsBatch], requires_grad=False)
        if cuda: x,y = x.cuda(), y.cuda()

        y_til = model(x)
        eval_loss = loss(y_til,y)

        acc += computeAccuracy(F.log_softmax(y_til, dim=0),y)
        l += eval_loss.data[0]
    return l/nb_iter,acc/nb_iter

if __name__ == '__main__':
    trainDataFile = "mnist_rotation_new/mnist_all_rotation_normalized_float_train_valid.amat"
    testDataFile = "mnist_rotation_new/mnist_all_rotation_normalized_float_test.amat"

    print('loading data...',end='',flush=True)
    x_train,y_train,x_test,y_test = load_data(trainDataFile, testDataFile)
    print('done',flush=True)

    #hyperparameters
    learning_rate = 0.001
    batch_size = 32
    train_size = x_train.shape[0]
    epochs_nb = 30
    cuda = False
    #WARNING - Task specific
    input_dim = 784
    output_dim = 10

    #shuffle train and test data
    perm = t.randperm(train_size)
    x_train = x_train[perm]
    y_train = y_train[perm]



    #Baseline DNN settings, according to paper
    model = baseDNN(input_dim, output_dim)

    model = DEN([784,500,500])
    for i in range(9):
        model.add_task()
    loss = nn.CrossEntropyLoss()

    if cuda:
        model = model.cuda()
        loss = loss.cuda()

    optim = t.optim.Adam(model.parameters(), lr=learning_rate)

    #book keeping
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    #training of model
    for e in range(epochs_nb):
        print('epoch '+str(e))
        model.batch_pass(x_train, y_train, loss, optim)

        #evaluation of current model
        train_l,train_acc = evaluation(model, x_train,y_train)
        test_l,test_acc = evaluation(model, x_test,y_test)

        train_losses.append(train_l)
        train_accuracies.append(train_acc)
        test_losses.append(test_l)
        test_accuracies.append(test_acc)

    plt.figure()
    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.legend(['train loss', 'test loss'], loc='lower right')

    plt.figure()
    plt.plot(train_accuracies)
    plt.plot(test_accuracies)
    plt.legend(['train accuracy', 'test accuracy'], loc='lower right')

    plt.show()
