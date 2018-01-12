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
import sklearn.metrics as metric

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
        return out

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



def oneHot(x, nbClass):
    xOneHot = np.zeros((len(x), nbClass))
    xOneHot[np.arange(len(x)), x] = 1
    return xOneHot


def evaluation(model, x_eval,y_eval,nb_class):
    l = 0
    nb_iter = x_eval.shape[0]//eval_batch_size
    outputs = []
    for k in range(nb_iter):
        #load batch
        indsBatch = range(k * eval_batch_size, (k+1) * eval_batch_size)
        x = Variable(x_eval[indsBatch, :], requires_grad=False)
        y = Variable(y_eval[indsBatch], requires_grad=False)
        if cuda: x,y = x.cuda(), y.cuda()

        y_til = model(x)
        eval_loss = loss(y_til,y)
        outputs += [F.log_softmax(y_til, dim=0)]
        l += eval_loss.data[0]

    auroc = metric.roc_auc_score(oneHot(y_eval[0:eval_batch_size*nb_iter].numpy(),nb_class),t.cat(outputs,0).data.numpy())
    return l/nb_iter,auroc

if __name__ == '__main__':
    trainDataFile = "mnist_rotation_new/mnist_all_rotation_normalized_float_train_valid.amat"
    testDataFile = "mnist_rotation_new/mnist_all_rotation_normalized_float_test.amat"

    print('loading data...',end='',flush=True)
    x_train,y_train,x_test,y_test = load_data(trainDataFile, testDataFile)
    print('done',flush=True)

    #hyperparameters
    learning_rate = 0.001
    batch_size = 32
    eval_batch_size = 2048
    train_size = x_train.shape[0]
    epochs_nb = 10
    cuda = False
    #WARNING - Task specific
    input_dim = 784
    output_dim = 10

    #shuffle train and test data
    perm = t.randperm(train_size)
    x_train = x_train[perm]
    y_train = y_train[perm]



    #DNN model as presented in paper
    model = baseDNN(input_dim, 2)
    '''
    model = DEN([784,500,200])
    print(model)
    print(model.depth)
    for l in model.layers:
    	print(l.weight.shape)
    model.add_neurons(1, 30)
    model.add_neurons(0, 30)
    print(model)
    for l in model.layers:
    	print(l.weight.shape)
    for i in range(9):
        model.add_task()
    '''
    loss = nn.CrossEntropyLoss()

    if cuda:
        model = model.cuda()
        loss = loss.cuda()

    optim = t.optim.Adam(model.parameters(), lr=learning_rate)

    #book keeping per task
    test_aurocs = []
    #overall book_keeping
    mean_auroc_test = []

    #training of binary model for each task from 1 to T
    task_y_train = t.LongTensor(y_train.shape).zero_()
    task_y_test = t.LongTensor(y_test.shape).zero_()
    for task_nb in range(output_dim):
        print("task " + str(task_nb))
        #create mapping for binary classif in oneVSall fashion
        task_y_train.zero_()
        task_y_test.zero_()
        task_y_train[y_train == task_nb] = 1
        task_y_test[y_test == task_nb] = 1

        #training of model on task
        for e in range(epochs_nb):
            print('epoch '+str(e))
            model.batch_pass(x_train, task_y_train, loss, optim)

            #evalution of auroc'score on test for this epoch
            _,test_auroc = evaluation(model, x_test,task_y_test,2)
            test_aurocs.append(test_auroc)

        mean_auroc_test.append((test_aurocs[-1] + sum(mean_auroc_test))/(len(mean_auroc_test) + 1))
        test_aurocs = []

    plt.figure()
    plt.plot(mean_auroc_test)
    plt.legend(['DNN'], loc='upper right')
    plt.xlabel("Number of tasks2")
    plt.ylabel("AUROC")

    plt.show(block=False)

