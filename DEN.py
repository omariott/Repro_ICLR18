import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import main_task_decomposition as helper
import numpy as np
import matplotlib.pyplot as plt

class DEN(nn.Module):
    def __init__(self, sizes,cuda=False):
        super(DEN, self).__init__()
        self.depth = len(sizes)
        self.sizes_hist = []
        self.num_tasks = 1
        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(self.depth-1)] + [nn.Linear(sizes[-1], 1)])
        self.w_hooks = [t.zeros(size) for size in sizes]
        self.b_hooks = [t.zeros(size) for size in sizes]
        self.hook_handles = []
        self.use_cuda = cuda


    def forward(self, x):
            for i, linear in enumerate(self.layers):
                if i<self.depth-1:
                    x = F.relu(linear(x))
                else: #output layer
                    out = linear(x)
            return out


    def param_norm(self, p=2):
        """
        Computes l-p norm of parameters
        """
        norm = 0
        for l in list(self.parameters()):
            norm += l.norm(p)
        return norm


    def group_norm(self, p=2):
        """
        Computes group norm for group regularization
        """
        norm = 0
        for l in self.layers:
            in_features = l.in_features
            coeff = in_features**(.5)
            norm += l.weight.norm(p) * coeff
        return norm


    def drift(self, old_params_list):
        """
        Computes drift from parameters at previous steps
        """
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


    def sparsify_clip(self, old_params_list):
    """
    Sparsify network with gradient cliping technique
    """
        for i, l in enumerate(self.parameters()):
            mask = (l*old_params_list[i]) > 0
            l.data *= mask.data.float()

    def sparsify_thres(self, tau=1e-2):
    """
    Sparsify network according to threshold tau
    """
        for i, l in enumerate(self.parameters()):
            mask = l.data.abs() > tau
            l.data *= mask.float()

    def sparsify_n_remove(self, nb_add_neurons, layer_ind, layer_params_masks, tau):
    """
    Sparsify part of a layer (according to given mask) and remove neurons with null incoming connections
    """
        #step 1 sparsify layer (only new weights)
        l_params = list(self.layers[layer_ind].parameters()) #returns [connections,biases]
        for i,weights in enumerate(l_params):
            sparsify_mask = weights.data.abs() > tau
            #ignore sparsify (= set to 1) on old weights (which are = to 0 in layer_mask)
            if(self.use_cuda):
                sparsify_mask[layer_params_masks[i].cuda() == 0] = 1
            else:
                sparsify_mask[layer_params_masks[i] == 0] = 1
            weights.data *= sparsify_mask.float()

        #step 2 remove useless neurons (only new neurons)

        #keep index of all neurons we will keep, starting by old ones
        old_nb_neurons = self.layers[layer_ind].out_features-nb_add_neurons
        idx_kept_neurons = list(range(0,old_nb_neurons))
        for i,n_connections in enumerate(l_params[0][old_nb_neurons:,:]):

            #WARNING DO NOT TAKE BIASES INTO ACCOUNT

            #if incoming weights of neurons are not all 0, keep it
            if not (n_connections.data.abs().sum() == 0):
                idx_kept_neurons.append(i)

        idx_kept_neurons = t.LongTensor(idx_kept_neurons)
        if self.use_cuda: idx_kept_neurons = idx_kept_neurons.cuda()
        #creates new layer
        new_layer = nn.Linear(self.layers[layer_ind].in_features,len(idx_kept_neurons))
        new_layer.weight.data = t.index_select(l_params[0].data,0,idx_kept_neurons)
        new_layer.bias.data = t.index_select(l_params[1].data,0,idx_kept_neurons)

        #adapt layer n+1 to new layer
        next_layer = nn.Linear(len(idx_kept_neurons),self.layers[layer_ind+1].out_features)
        old_next_l_params = list(self.layers[layer_ind+1].parameters())

        next_layer.weight.data = t.index_select(old_next_l_params[0].data,1,idx_kept_neurons)
        next_layer.bias.data = old_next_l_params[1].data

        if self.use_cuda: new_layer,next_layer = new_layer.cuda(),next_layer.cuda()
        self.layers[layer_ind] = new_layer
        self.layers[layer_ind+1] = next_layer

    def sparsity(self):
    """
    Computes sparsity of network (proportion of zeroed parameters)
    """
        num = 0
        denom = 0
        for i, l in enumerate(self.parameters()):
            num += (l == 0).float().sum().data[0]
            prod = 1
            for dim in l.size():
                prod *= dim
            denom += prod
        return num/denom


    def add_task(self):
    """
    Adds a new output neuron to the network
    """
        # WARNING Probably kills cuda
        self.num_tasks += 1
        #register sizes for inference
        self.sizes_hist.append([l.in_features for l in self.layers])
        #add output neuron
        old_output = self.layers[-1]
        new_output = self.add_output_dim(old_output)
        self.layers[self.depth-1] = new_output


    def add_neurons(self, l, n_neurons=1):
    """
    Adds n_neurons new neuros to layer l of the network
    Used during dynamic expansion
    """
        # add neurons to layer number l
        if l > (self.depth - 1):
            print("Error, trying to add neuron to output layer. Please use 'add_task' method instead")
            exit(-1)
        #add neurons to layer l
        old_layer = self.layers[l]
        new_layer = self.add_output_dim(old_layer, n_neurons)
        self.layers[l] = new_layer

        #add connections to layer l+1
        old_layer = self.layers[l+1]
        new_layer = self.add_input_dim(old_layer, n_neurons)
        self.layers[l+1] = new_layer


    def copy_neuron(self, layer_index, connections, bias):
    """
    creates a duplicate of a neuron
    used during network duplication
    """
        # add neuron and copy connection weights
        self.add_neurons(layer_index)
        self.layers[layer_index].weight[-1].data = connections.data
        self.layers[layer_index].bias[-1].data = bias.data

    def compute_hooks(self):
    """
    creates backwar hooks for training specific neurons
    """
        current_layer = self.depth-1
        # mask of selected neurons for output layer, we only get the last one corresponding to the new tasks
        out_mask = t.zeros(self.num_tasks)
        out_mask[-1] = 1
        while(current_layer >= 0):
            #get the weights between current layer and the following one
            connections = self.layers[current_layer].weight.data
            output_size, input_size = connections.shape
            in_mask = t.zeros(input_size)
            for index, line in enumerate(connections):
                if(out_mask[index] == 1):
                    t.max(in_mask, (line.cpu() != 0).float(), out=in_mask)
            if self.use_cuda:
                self.b_hooks[current_layer] = out_mask.cuda()
                self.w_hooks[current_layer] = t.mm(out_mask.unsqueeze(1), in_mask.unsqueeze(0)).cuda()
            else:
                self.b_hooks[current_layer] = out_mask
                self.w_hooks[current_layer] = t.mm(out_mask.unsqueeze(1), in_mask.unsqueeze(0))
            out_mask = in_mask
            current_layer -= 1

    def swap_neuron(self, layer, old_n, new_n):
    """
    swaps the positions of two neurons
    """
        #incomming weights
        l = self.layers[layer]
        #save weights
        buff_w = l.weight.data[old_n]
        buff_b = l.bias.data[old_n]
        #copy new to old
        l.weight.data[old_n] = l.weight.data[new_n]
        l.bias.data[old_n] = l.bias.data[new_n]
        #copy new to old
        l.weight.data[new_n] = buff_w
        l.bias.data[new_n] = buff_b
        #outgoing weights
        l = self.layers[layer+1]
        buff_w = l.weight.data[:,old_n]
        l.weight.data[:,old_n] = l.weight.data[:,new_n]
        l.weight.data[:,new_n] = buff_w



    def register_hooks(self):
    """
    registers bacward hooks for next training session
    """
        for i, l in enumerate(self.layers):
            self.hook_handles.append(l.bias.register_hook(make_hook(self.b_hooks[i])))
            self.hook_handles.append(l.weight.register_hook(make_hook(self.w_hooks[i])))

    def unhook(self):
    """
    removes backward hooks
    """
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def batch_pass(self, x_train, y_train, loss, optim, mu=0.1, batch_size=32, reg_list=None, args_reg=None):
        #incremental learning batch pass (output considered dependant on task)
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
            """
            if reg_list is not None:
                for i,r in enumerate(reg_list):
                    l += mu * r(*args_reg[i])
            """
            optim.zero_grad()
            l.backward()
            optim.step()


        start_val = (i+1) * batch_size

        if reg_list is not None:
            optim.zero_grad()
            l=0
            for i,r in enumerate(reg_list):
                l += mu * r(*args_reg[i])
            l.backward()
            optim.step()


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

    def selective_retrain(self, x_train, y_train, loss, optimizer, n_epochs=10, mu=.1):
        # Retrain output layer
        out_params = self.layers[-1].parameters()
        output_optimizer = t.optim.SGD(out_params, lr=0.01)
        old_l = float('inf')
        for e in range(n_epochs):
            l = self.batch_pass(x_train, y_train, loss, output_optimizer, mu=mu, reg_list=[self.param_norm], args_reg=[[1]])
            #Early stopping
            if(old_l - l < 1e-3):
                print("SR_last:", e,"epochs")
                break
            old_l = l
        self.sparsify_thres()
        # perform BFS
        self.compute_hooks()
        self.register_hooks()



        # train subnetwork
        #init book-keeping
        train_losses = []
        train_accs = []
        old_l = float('inf')
        for e in range(n_epochs):
            l = self.batch_pass(x_train, y_train, loss, optimizer, mu=mu, reg_list=[self.param_norm], args_reg=[[2]])
            #Early stopping
            if(old_l - l < 0):
                print("SR:", e,"epochs")
                break
            old_l = l
            #eval network's loss and acc
            train_l,_,train_acc = helper.evaluation(self, loss, x_train, y_train, 2,use_cuda=self.use_cuda)
            train_accs.append(train_acc)
            train_losses.append(train_l)

        self.unhook()
        return train_losses[-1]

    def dynamic_expansion(self, x_train, y_train, loss, retrain_loss, tau=0.02, n_epochs=10, mu=0.1): #tau was 0.02
        nb_add_neuron = 10
        learning_rate = 0.01
        sparse_thr = 0.01

        #if given loss isn't low enough, expand network
        if (retrain_loss > tau):

            #add new units
            for l in range(self.depth-1):
                self.add_neurons(l,nb_add_neuron)
            #train newly added neurons


            #first register hook for each layer
            for i,l in enumerate(self.layers):
                #define hook depending on considered layer
                if i == 0:
                    def my_hook(grad):
                        grad[:-nb_add_neuron,:] = 0
                        return grad
                elif i == self.depth-1:
                    def my_hook(grad):
                        grad[:,:-nb_add_neuron] = 0
                        return grad
                else: #hidden layers
                    def my_hook(grad):
                        grad[:-nb_add_neuron,:-nb_add_neuron] = 0
                        return grad

                #register hook to weight variable
                self.hook_handles.append(l.weight.register_hook(my_hook))

                #take care of bias
                if i == self.depth-1:
                    continue
                else: #hidden layers
                    def my_hook(grad):
                        grad[:-nb_add_neuron] = 0
                        return grad
                #register hook to bias variable
                self.hook_handles.append(l.bias.register_hook(my_hook))



            #train added neurons with l1 norm for sparsity
            optimizer = t.optim.SGD(self.parameters(), lr=learning_rate)
            old_l = float('inf')
            for e in range(n_epochs):
                l = self.batch_pass(x_train, y_train, loss, optimizer, mu=mu, reg_list=[self.group_norm,self.param_norm], args_reg=[[2],[1]])
                #Early stopping
                if(old_l - l < 0):
                    print("NE:", e,"epochs")
                    break
                old_l = l
            self.unhook()


            #sparsify, layer-wise
            for i in range(self.depth-1):
                connections, biases = self.layers[i].parameters()
                c_new_neurons_mask = t.ones(connections.shape).byte()
                b_new_neurons_mask = t.ones(biases.shape).byte()
                if i == 0:
                    c_new_neurons_mask[:-nb_add_neuron,:] = 0
                    b_new_neurons_mask[:-nb_add_neuron] = 0

                else: #hidden layers
                    c_new_neurons_mask[:-nb_add_neuron,:-nb_add_neuron] = 0
                    b_new_neurons_mask[:-nb_add_neuron] = 0

                self.sparsify_n_remove(nb_add_neuron, i, [c_new_neurons_mask,b_new_neurons_mask], sparse_thr)



        else:
            print("loss:" + str(retrain_loss) + ",low enough, dynamic_expansion not required")


    def duplicate(self, x_train, y_train, loss, optimizer, old_params_list, n_epochs=20, sigma=.2, lambd=10): #sigma was .002
        # Retrain network once again
        optimizer = t.optim.SGD(self.parameters(), lr=0.01)

        old_l = float('inf')
        for e in range(n_epochs):
            l = self.batch_pass(x_train, y_train, loss, optimizer, mu=lambd, reg_list=[self.drift], args_reg=[[old_params_list]])
            if(old_l - l < 0):
                print("Split1:", e,"epochs")
                break
            old_l = l

        # Compute connection-wise distance
        splits_index = []
        for num_layer,layer in enumerate(self.layers):
            splits = []
            if(num_layer == self.depth-1):
                break ##Exiting loop on output layer
            old_layer = old_params_list[2*num_layer]
            old_bias = old_params_list[2*num_layer+1]
            old_shape = old_layer.shape
            new_layer = layer.weight[:old_shape[0], :old_shape[1]]
            new_bias = layer.bias[:old_shape[0]]
            # If distance > sigma, add old neuron to network
            for num_neuron, old_neuron in enumerate(old_layer):
                connection_drift = (old_neuron - new_layer[num_neuron]).norm(2).data[0]
                bias_drift = (old_bias[num_neuron] - new_bias[num_neuron]).norm(2).data[0]
                if (connection_drift + bias_drift > sigma):
                    self.copy_neuron(num_layer, old_neuron, old_bias[num_neuron])
                    splits.append(num_neuron)
            splits_index.append(splits)

        # Retrain
        optimizer = t.optim.SGD(self.parameters(), lr=0.01)
        old_l = float('inf')
        for e in range(n_epochs):
            l = self.batch_pass(x_train, y_train, loss, optimizer, mu=lambd, reg_list=[self.drift], args_reg=[[old_params_list]])
            if(old_l - l < 0):
                print("Split2:", e,"epochs")
                break
            old_l = l
        # Obselete : Real algo doesn't use this
        """
        #Restore old neurons
        for num_layer, layer in enumerate(self.layers):
            if(num_layer == self.depth-1):
                break ##Exiting loop on output layer
            old_shape_out, old_shape_in = old_params_list[2*num_layer].shape
            for index, n_index in enumerate(splits_index[num_layer]):
                self.swap_neuron(num_layer, n_index, old_shape_out+index)

        """

    def create_eval_model(self, task_num):
    """
    creates truncated models according to timestamps for evaluation
    """
        inference_sizes = self.sizes_hist[task_num]
        eval_model = DEN(inference_sizes, cuda=self.use_cuda)
        for i in range(task_num):
            eval_model.add_task()
        for index_l, l in enumerate(eval_model.layers):
            in_features = l.in_features
            out_features = l.out_features
            l.weight.data = self.layers[index_l].weight.data[:out_features,:in_features]
            l.bias.data = self.layers[index_l].bias.data[:out_features]
        return eval_model



    def add_output_dim(self, old_layer, n_neurons=1):
        """
        adds neurons to a layer
        """
#        old_mat = old_layer.weight.data

        input_dim, output_dim = old_layer.in_features, old_layer.out_features
        if old_layer.bias is not None:
            new_layer = nn.Linear(input_dim, output_dim + n_neurons)
            new_layer.bias.data[:-n_neurons] = old_layer.bias.data
        else:
            new_layer = nn.Linear(input_dim, output_dim + n_neurons, bias=False)
        new_layer.weight.data[:-n_neurons,:] = old_layer.weight.data
        if self.use_cuda: new_layer = new_layer.cuda()

#        diff = old_mat - new_layer.weight.data[:-n_neurons]
#        plt.imshow(diff.float().numpy(), cmap='hot')
#        plt.colorbar()
#        plt.show()

        return new_layer


    def add_input_dim(self, old_layer, n_neurons=1):
        """
        adds connections to a layer to accomodate
        for new neurons in the previous layer
        """
        # WARNING Probably kills cuda
        input_dim, output_dim = old_layer.in_features, old_layer.out_features
        if old_layer.bias is not None:
            new_layer = nn.Linear(input_dim + n_neurons, output_dim)
        else:
            new_layer = nn.Linear(input_dim + n_neurons, output_dim, bias=False)
        new_layer.weight.data[:,:-n_neurons] = old_layer.weight.data
        if self.use_cuda: new_layer = new_layer.cuda()
        return new_layer


def make_hook(hook):
    """
    create a hook to be registered
    """
    def hooker(grad):
        return grad * Variable(hook, requires_grad=False)$
    return hooker
