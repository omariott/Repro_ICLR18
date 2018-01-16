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


def computeAccuracy(y_out, y_true):
    #pred and y are n*1 numpy arrays
    preds = np.zeros(np.shape(y_true))
    preds[y_out.squeeze() >= 0.5] = 1
    accuracy = metric.accuracy_score(y_true,preds)
    return accuracy


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

        y_til = model(x)[:,(model.num_tasks-1)]
        out = F.sigmoid(y_til)
        eval_loss = loss(out,y)
        outputs += [out]
        l += eval_loss.data[0]

    np_y_eval = y_eval[0:eval_batch_size*nb_iter].int().numpy()
    y_score = t.cat(outputs,0).data.numpy()
    auroc = metric.roc_auc_score(np_y_eval,y_score)
    acc = computeAccuracy(y_score,np_y_eval)
    return l/nb_iter,auroc,acc

if __name__ == '__main__':
    #trainDataFile = "mnist_rotation_new/mnist_all_rotation_normalized_float_train_valid.amat"
    #testDataFile = "mnist_rotation_new/mnist_all_rotation_normalized_float_test.amat"
    trainDataFile = "mnist/mnist_train.amat"
    testDataFile = "mnist/mnist_test.amat"

    print('loading data...',end='',flush=True)
    x_train,y_train,x_test,y_test = load_data(trainDataFile, testDataFile)
    print('done',flush=True)

    #hyperparameters
    learning_rate = 0.01
    batch_size = 32
    eval_batch_size = 2048
    train_size = x_train.shape[0]
    epochs_nb = 5
    cuda = False
    verbose = True
    #WARNING - Task specific
    input_dim = 784
    output_dim = 10

    #shuffle train and test data
    perm = t.randperm(train_size)
    x_train = x_train[perm]
    y_train = y_train[perm]


    is_DEN = True

    #DNN model as presented in paper
    if is_DEN:
        model = DEN([784,500,500])
    else:
        #WARNING NO LONGER WORKS
        model = baseDNN(input_dim, 2)

    loss = nn.BCELoss()

    if cuda:
        model = model.cuda()
        loss = loss.cuda()

    optim = t.optim.SGD(model.parameters(), lr=learning_rate)

    #book keeping per task
    test_aurocs = []
    train_aurocs = []
    train_losses = []
    test_losses = []
    test_accs = []
    train_accs = []
    #overall book_keeping
    tasks_test_aurocs = []
    tasks_train_aurocs = []

    #training of binary model for each task from 1 to T
    task_y_train = t.FloatTensor(y_train.shape).zero_()
    task_y_test = t.FloatTensor(y_test.shape).zero_()
    for task_nb in range(4):
        print("task " + str(task_nb))
        #create mapping for binary classif in oneVSall fashion
        task_y_train.zero_()
        task_y_test.zero_()
        task_y_train[y_train == task_nb] = 1
        task_y_test[y_test == task_nb] = 1

        #training of model on task
        if(model.num_tasks == 1):
            for e in range(epochs_nb):
#                print('epoch '+str(e))
                model.batch_pass(x_train, task_y_train, loss, optim)
        else:
            model.selective_retrain(x_train, y_train, loss, optim, n_epochs=epochs_nb)

        """
        À réparer !!!
        """
        for e in range(epochs_nb):
#            print('epoch '+str(e))
#            model.batch_pass(x_train, task_y_train, loss, optim)

            #evaluation of auroc'score for this epoch
            test_l,test_auroc,test_acc = evaluation(model, x_test,task_y_test,2)
            train_l,train_auroc,train_acc = evaluation(model, x_train,task_y_train,2)
            test_aurocs.append(test_auroc)
            train_aurocs.append(train_auroc)
            test_losses.append(test_l)
            train_losses.append(train_l)
            test_accs.append(test_acc)
            train_accs.append(train_acc)

        tasks_test_aurocs.append(test_aurocs[-1])
        tasks_train_aurocs.append(train_aurocs[-1])

        model.add_task()

        if verbose:
            plt.figure()
            plt.plot(train_losses)
            plt.plot(test_losses)
            plt.legend(['train loss','test loss'], loc='upper right')
            plt.xlabel("nb of epochs")
            plt.ylabel("loss")
            plt.show(block=False)
#            fig.savefig("loss"+str(task_nb))

            fig = plt.figure()
            plt.plot(train_accs)
            plt.plot(test_accs)
            plt.legend(['train accuracy','test accuracy'], loc='upper right')
            plt.xlabel("nb of epochs")
            plt.ylabel("accuracy")
            plt.show(block=False)
#            fig.savefig("acc"+str(task_nb))

        test_aurocs = []
        train_aurocs = []
        test_accs = []
        train_accs = []
        train_losses = []
        test_losses = []

    fig = plt.figure()
    plt.plot(tasks_train_aurocs)
    plt.plot(tasks_test_aurocs)
    plt.legend(['DEN auroc train','DEN auroc test'], loc='upper right')
    plt.xlabel("Number of tasks")
    plt.ylabel("AUROC")
    plt.show(block=False)
#    fig.savefig("ROC")
