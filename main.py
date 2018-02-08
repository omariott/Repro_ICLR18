# -*- coding: utf-8 -*-
"""
ICLR Reproducibility challenge 2018
Code reproduction of https://arxiv.org/pdf/1708.01547.pdf
@author: Octave Mariotti & Rémy Portelas
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
import DEN as DEN_model


"""
DNN-STL is a base deep neural network trained for each task separately
It will use t BaseDNN networks traned on each task
"""
class DNN_STL(nn.Module):
    def __init__(self,sizes,mu=0.1,cuda=False):
        self.models = [baseDNN([784,312,128],mu=mu,cuda=cuda) for i in range(10)]
        self.num_tasks = 1

    def add_task(self):
        self.num_tasks += 1
        #set new model according to the task it will work on
        if self.num_tasks < 11: self.models[self.num_tasks-1].num_tasks = self.num_tasks

    def batch_pass(self, x_train, y_train, loss, optim, mu=0.1,batch_size=32, reg_list=None, args_reg=None):
        return self.models[self.num_tasks-1].batch_pass(x_train, y_train, loss, optim, mu, batch_size, reg_list, args_reg)

    def forward(self, x):
        return self.models[self.num_tasks-1].forward(x)

    def parameters(self):
        return self.models[self.num_tasks-1].parameters()

    def create_eval_model(self, task_num):
        return self.models[task_num]

    def sparsity(self):
        return 0.

    def sparsify_thres(self, tau=0.01):
        pass

    def param_norm(self, p=2):
        return self.models[self.num_tasks-1].param_norm(p)

"""
This class defines a simble deep neural network.
It can use semantic drift regularization (DNN-L2 in paper) or l-2 regularization (DNN in paper)
"""
class baseDNN(nn.Module):

    def __init__(self,sizes,mu=0.1,cuda=False):
        nn.Module.__init__(self)
        self.depth = len(sizes)
        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(self.depth-1)] + [nn.Linear(sizes[-1], 10)])
        self.use_cuda = cuda
        self.num_tasks = 1

    def add_task(self):
        self.num_tasks += 1

    def param_norm(self, p=2):
        norm = 0
        for l in list(self.parameters()):
            norm += l.norm(p)
        return norm

    def drift(self, old_params_list):
        norm = 0
        cur_params_list = list(self.parameters())
        for l1,l2 in zip(old_params_list, cur_params_list):
            #Get old shape
            old_shape = l1.shape
            #extract new params according to shape
            if len(old_shape) == 1:
                new_layer = l2[:old_shape[0]]
            else:
                new_layer = l2[:old_shape[0], :old_shape[1]]
            norm += (l1-new_layer).norm(2)
        return norm

    def create_eval_model(self, task_num):
        return self

    def forward(self, x):
        for i, linear in enumerate(self.layers):
            if i<self.depth-1:
                x = F.relu(linear(x))
            else: #output layer
                out = linear(x)
        return out

    def batch_pass(self, x_train, y_train, loss, optim, mu=0.1,batch_size=32, reg_list=None, args_reg=None):
        set_size = x_train.shape[0]
        split = 5/6
        train_size = int(set_size * split)
        val_size = set_size - train_size

        for i in range(train_size // batch_size):
            #load batch
            indsBatch = range(i * batch_size, (i+1) * batch_size)
            x = Variable(x_train[indsBatch, :], requires_grad=False)
            y = Variable(y_train[indsBatch], requires_grad=False)
            if self.use_cuda: x,y = x.cuda(), y.cuda()

            #forward
            y_til = self.forward(x)[:,(self.num_tasks-1)]
            #loss and backward
            l = loss(F.sigmoid(y_til),y.float())
            optim.zero_grad()
            l.backward()
            optim.step()

        optim.zero_grad()
        l=0
        for i,r in enumerate(reg_list):
            l += mu * r(*args_reg[i])
        l.backward()
        optim.step()

        start_val = (i+1) * batch_size

        l = 0
        for i in range(val_size // batch_size):
            #load batch
            indsBatch = range(start_val + i * batch_size, start_val + (i+1) * batch_size)
            x = Variable(x_train[indsBatch, :], requires_grad=False)
            y = Variable(y_train[indsBatch], requires_grad=False)
            if self.use_cuda: x,y = x.cuda(), y.cuda()

            #forward
            y_til = self.forward(x)[:,(self.num_tasks-1)]
            #loss and backward
            l += loss(F.sigmoid(y_til),y.float())/batch_size
        return l.data[0]

    def sparsity(self):
        return 0.0

    def sparsify_thres(self, tau=0.01):
        pass


def read_data(fname):
    f = open(fname)
    f.readline()  # skip the header
    data = np.loadtxt(f)
    x = data[:, :-1]
    y = data[:, -1].astype(int)
    return x,y

#load data from given files and save it as tensors to speed up next execution
def load_data(trainDataFile,testDataFile):
    if not os.path.isfile('data_tensors.p'):
        x_train, y_train = read_data(trainDataFile)
        x_test, y_test = read_data(testDataFile)
        nb_class = len(np.unique(y_test))

        x_train = t.FloatTensor(x_train)
        y_train = t.LongTensor(y_train)
        x_test = t.FloatTensor(x_test)
        y_test = t.LongTensor(y_test)

        data = [x_train,y_train,x_test,y_test]
        pickle.dump(data, open('data_tensors.p', 'wb'))
        return data
    else:
        return pickle.load(open('data_tensors.p', 'rb'))

def computeAccuracy(y_out, y_true):
    #pred and y are n*1 numpy arrays
    preds = np.zeros(np.shape(y_true))
    preds[y_out.squeeze() >= 0.5] = 1
    accuracy = metric.accuracy_score(y_true,preds)
    return accuracy

"""
Evaluates given model on given data,
RETURNS: loss, area under ROC curve (auroc) and accuracy
"""
def evaluation(model,loss,x_eval,y_eval,nb_class,use_cuda=False):
    eval_batch_size = 2048
    l = 0
    nb_iter = x_eval.shape[0]//eval_batch_size
    outputs = []
    for k in range(nb_iter):
        #load batch
        indsBatch = range(k * eval_batch_size, (k+1) * eval_batch_size)
        x = Variable(x_eval[indsBatch, :], requires_grad=False)
        y = Variable(y_eval[indsBatch], requires_grad=False)

        if use_cuda:
            x,y = x.cuda(),y.cuda()

        y_til = model.forward(x)[:,(model.num_tasks-1)]
        out = F.sigmoid(y_til)
        eval_loss = loss(out,y)
        outputs += [out]
        l += eval_loss.data[0]

    np_y_eval = y_eval[0:eval_batch_size*nb_iter].int().cpu().numpy()
    y_score = t.cat(outputs,0).data.cpu().numpy()
    auroc = metric.roc_auc_score(np_y_eval,y_score)
    acc = computeAccuracy(y_score,np_y_eval)
    return l/nb_iter,auroc,acc

#evaluate our model for all learned tasks, after learning is complete. y_eval is a 1D vector with values from 0 to task-1
def overall_offline_evaluation(model, loss, x_eval,y_eval,use_cuda=False):
    eval_batch_size = 2048
    nb_tasks = model.num_tasks - 1
    timestamped_models = [model.create_eval_model(t_idx) for t_idx in range(nb_tasks)]
    aurocs = np.zeros(nb_tasks)
    accs = np.zeros(nb_tasks)
    nb_iter = x_eval.shape[0]//eval_batch_size
    outputs = {t_idx:[] for t_idx in range(nb_tasks)}
    for k in range(nb_iter):
        #load batch
        indsBatch = range(k * eval_batch_size, (k+1) * eval_batch_size)
        x = Variable(x_eval[indsBatch, :], requires_grad=False)
        y = Variable(y_eval[indsBatch], requires_grad=False)

        if use_cuda:
            x,y = x.cuda(),y.cuda()

        y_til = [model(x) for model in timestamped_models]

        for t_idx in range(nb_tasks):
            y_til_t = y_til[t_idx]
            out = F.sigmoid(y_til_t[:,t_idx])
            outputs[t_idx] += [out]

    task_y_eval = t.FloatTensor(y_eval.shape).zero_()
    for t_idx in range(nb_tasks):
        #create mapping for binary classif in oneVSall fashion
        task_y_eval.zero_()
        task_y_eval[y_eval == t_idx] = 1

        np_y_eval = task_y_eval[0:eval_batch_size*nb_iter].int().cpu().numpy()
        y_score = t.cat(outputs[t_idx],0).data.cpu().numpy()
        aurocs[t_idx] = metric.roc_auc_score(np_y_eval,y_score)
        accs[t_idx] = computeAccuracy(y_score,np_y_eval)
    return accs,aurocs


def plot_curves(data_lists,model_name,curve_type,x_axis='nb of epochs',save_plot=True,display_plot=False,savedir="./figures/",filename='lonely_plot',styles=None,curve_names=['train','test']):
    #data_lists must contain [train,test] or [train] values
    fig = plt.figure()
    for i,values in enumerate(data_lists):
        if styles == None:
            plt.plot(range(1,len(values)+1), values)
        else:
            plt.plot(range(1,len(values)+1), values,styles[i])
    if len(data_lists) == 2:
        plt.legend([curve_names[0]+curve_type,curve_names[1]+curve_type], loc='upper right')
    else:#only train
        plt.legend([curve_names[1]+curve_type], loc='upper right')
    plt.xlabel(x_axis)
    plt.ylabel(curve_type)
    if save_plot:
        fig.savefig(savedir+filename)
    if display_plot:
        plt.show(block=False)


if __name__ == '__main__':
    #datasets available here: http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/MnistVariations
    #trainDataFile = "mnist/mnist_train.amat"
    #testDataFile = "mnist/mnist_test.amat"
    trainDataFile = "mnist_all_background_images_rotation_normalized_train_valid.amat"
    testDataFile = "mnist_all_background_images_rotation_normalized_test.amat"

    savedir = "./figures"
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    print('loading data...',end='',flush=True)
    x_train,y_train,x_test,y_test = load_data(trainDataFile, testDataFile)
    print('done',flush=True)

    #hyperparameters
    nb_tasks = 10
    learning_rate = 0.01
    batch_size = 32
    train_size = x_train.shape[0]
    epochs_nb = 2
    cuda = False
    verbose = True
    input_dim = 784
    output_dim = 10

    #shuffle train and test data
    perm = t.randperm(train_size)
    x_train = x_train[perm]
    y_train = y_train[perm]


    model_type = "DEN" # DEN | DNN | DNN-L2 | DNN-STL

    #DNN model as presented in paper
    if model_type == "DEN":
        model = DEN_model.DEN([784,312,128],cuda=cuda)
    elif model_type == "DNN-STL":
        model = DNN_STL([784,312,128],mu=0.1,cuda=cuda)
    else: #DNN or DNN-L2
        model = baseDNN([784,312,128],mu=0.1,cuda=cuda)

    loss = nn.BCELoss()

    if cuda:
        model = model.cuda()
        loss = loss.cuda()

    if model_type == 'DEN': 
        print("model before training")
        print(model)

    #overall book_keeping
    test_aurocs = []
    train_aurocs = []
    test_average_aurocs_task = []

    #training of binary model for each task from 1 to T
    task_y_train = t.FloatTensor(y_train.shape).zero_()
    task_y_test = t.FloatTensor(y_test.shape).zero_()
    for task_nb in range(nb_tasks):
        print("task " + str(task_nb))
        #build optim
        optimizer = t.optim.SGD(model.parameters(), lr=learning_rate)
        #create mapping for binary classif in oneVSall fashion
        task_y_train.zero_()
        task_y_test.zero_()
        task_y_train[y_train == task_nb] = 1
        task_y_test[y_test == task_nb] = 1

        #Saving parameters for network split/duplication or DNN-L2 drift reg.
        old_params_list = [Variable(w.data.clone(), requires_grad=False) for w in model.parameters()]

        #training of model on task
        if(model_type != "DEN" or model.num_tasks == 1):
            old_l = float('inf')

            for e in range(epochs_nb):
                if model_type == "DEN":
                    l = model.batch_pass(x_train, task_y_train, loss, optimizer, mu=.01, reg_list=[model.param_norm], args_reg=[[1]])
                    model.sparsify_thres()
                elif model_type == "DNN":
                    l = model.batch_pass(x_train, task_y_train, loss, optimizer, reg_list=[model.param_norm], args_reg=[[2]])
                elif model_type == "DNN-L2":
                    l = model.batch_pass(x_train, task_y_train, loss, optimizer, reg_list=[model.drift], args_reg=[[old_params_list]])
                elif model_type == "DNN-STL":
                    l = model.batch_pass(x_train, task_y_train, loss, optimizer, reg_list=[model.param_norm], args_reg=[[2]])

                #Early stopping
                if(old_l - l < 0):
                    print("First train:", e,"epochs")
                    break
                old_l = l

        else:
            #Saving parameters for network split/duplication
            old_params_list = [Variable(w.data.clone(), requires_grad=False) for w in model.parameters()]
            #Selective retrain
            retrain_loss = model.selective_retrain(x_train, task_y_train, loss, optimizer, n_epochs=epochs_nb)
            #Network expansion
            model.dynamic_expansion(x_train, task_y_train, loss, retrain_loss, n_epochs=epochs_nb)
            #split
            model.duplicate(x_train, task_y_train, loss, optimizer, old_params_list, n_epochs=epochs_nb)

        #evaluation of auroc'score
        _,test_auroc,_ = evaluation(model, loss, x_test, task_y_test, 2, use_cuda=cuda)
        _,train_auroc,_ = evaluation(model, loss, x_train, task_y_train, 2, use_cuda=cuda)
        test_aurocs.append(test_auroc)
        train_aurocs.append(train_auroc)
        print("test_auroc: " + str(test_auroc))
        print("\n##################################\n")

        model.add_task()
        _,aurocs_test = overall_offline_evaluation(model, loss, x_test, y_test, use_cuda=cuda)
        test_average_aurocs_task.append(np.mean(aurocs_test))

    if model_type == 'DEN': 
        print("model after training")
        print(model)

    #online evaluation plot
    plot_curves([train_aurocs,test_aurocs],'DEN','auroc',x_axis='nb of tasks', filename="online_auroc",styles=['--rv','--bs'])

    #offline evaluation for all tasks
    _,aurocs_train = overall_offline_evaluation(model, loss, x_train, y_train, use_cuda=cuda)
    _,aurocs_test = overall_offline_evaluation(model, loss, x_test, y_test, use_cuda=cuda)
    plot_curves([aurocs_train,aurocs_test],'DEN','auroc',x_axis='nb of tasks', filename="offline_auroc",styles=['--rv','--bs'])
    plot_curves([test_average_aurocs_task],'DEN','auroc',x_axis='nb of tasks', filename="paper eval on test set",styles=['--rv'],curve_names=['test'])
