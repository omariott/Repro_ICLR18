import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import main_task_decomposition as helper

class DEN(nn.Module):
    def __init__(self, sizes):
        super(DEN, self).__init__()
        self.depth = len(sizes)
        self.sizes = sizes
        self.num_tasks = 1
        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(self.depth-1)] + [nn.Linear(sizes[-1], 1)])
        self.w_hooks = [t.zeros(size) for size in sizes]
        self.b_hooks = [t.zeros(size) for size in sizes]
        self.hook_handles = []

    def forward(self, x):
            for i, linear in enumerate(self.layers):
                if i<self.depth-1:
                    x = F.relu(linear(x))
                else: #output layer
                    out = linear(x)
            return out


    def param_norm(self, p=2):
        norm = 0
        for l in list(self.parameters()):
            norm += l.norm(p)
        return norm


    def group_norm(self, p=2):
        norm = 0
        for l in self.layers:
            in_features = l.in_features
            coeff = in_features**(.5)
            norm += l.weight.data.norm(p)
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


    def sparsify_clip(self, old_params_list):
        for i, l in enumerate(self.parameters()):
            mask = (l*old_params_list[i]) > 0
            l.data *= mask.data.float()

    def sparsify_thres(self, tau=0.01):
        for i, l in enumerate(self.parameters()):
            mask = l.data.abs() > tau
            l.data *= mask.float()

    def sparsity(self):
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
        # WARNING Probably kills cuda
        self.num_tasks += 1
        #add output neuron
        old_output = self.layers[-1]
        new_output = add_output_dim(old_output)
        self.layers[self.depth-1] = new_output


    def add_neurons(self, l, n_neurons=1):
        # WARNING Probably kills cuda
        # add neurons to layer number l
        if l > (self.depth - 1):
            print("Error, trying to add neuron to output layer. Please use 'add_task' method instead")
            exit(-1)
        #add neurons to layer l
        old_layer = self.layers[l]
        new_layer = add_output_dim(old_layer, n_neurons)
        self.layers[l] = new_layer

        #add connections to layer l+1
        old_layer = self.layers[l+1]
        new_layer = add_input_dim(old_layer, n_neurons)
        self.layers[l+1] = new_layer


    def copy_neuron(self, layer_index, connections, bias):
        # WARNING Probably kills cuda
        # add neuron and copy connection weights
        self.add_neurons(layer_index)
        self.layers[layer_index].weight[-1].data = connections.data
        self.layers[layer_index].bias[-1].data = bias.data

    def compute_hooks(self):
        current_layer = self.depth-1
        #mask of selected neurons for output layer, we only get the last one corresponding to the new tasks
        out_mask = t.zeros(self.num_tasks)
        out_mask[-1] = 1
        while(current_layer >= 0):
            #get the weights between current layer and the following one
            connections = self.layers[current_layer].weight.data
            output_size, input_size = connections.shape
            in_mask = t.zeros(input_size)
            for index, line in enumerate(connections):
                if(out_mask[index] == 1):
                    t.max(in_mask, (line != 0).float(), out=in_mask)
            self.b_hooks[current_layer] = out_mask
            self.w_hooks[current_layer] = t.mm(out_mask.unsqueeze(1), in_mask.unsqueeze(0))
            out_mask = in_mask
            current_layer -= 1

    def register_hooks(self):
        for i, l in enumerate(self.layers):
            self.hook_handles.append(l.bias.register_hook(make_hook(self.b_hooks[i])))
            self.hook_handles.append(l.weight.register_hook(make_hook(self.w_hooks[i])))

    def unhook(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def batch_pass(self, x_train, y_train, loss, optim, mu=0.1, batch_size=32, reg=None, args_reg=None, cuda=False):
        #incremental learning batch pass (output considered dependant on task)
        #print(list(self.parameters()))
        train_size = x_train.shape[0]
        for i in range(train_size // batch_size):
            #load batch
            indsBatch = range(i * batch_size, (i+1) * batch_size)
            x = Variable(x_train[indsBatch, :], requires_grad=False)
            y = Variable(y_train[indsBatch], requires_grad=False)
            if cuda: x,y = x.cuda(), y.cuda()

            #forward
            y_til = self.forward(x)[:,(self.num_tasks-1)]
            #loss and backward
            #print(F.sigmoid(y_til))
            l = loss(F.sigmoid(y_til),y.float())
            if reg is not None:
                l += mu * reg(*args_reg)
            optim.zero_grad()
            l.backward()
            optim.step()


    def selective_retrain(self, x_train, y_train, loss, optimizer, n_epochs=10, mu=0.1):
        """
            Retrain output layer
        """
        #Solving for output layer
        out_params = self.layers[-1].parameters()
        output_optimizer = t.optim.SGD(out_params, lr=0.1, weight_decay=0)
        # train it
        for i in range(n_epochs):
            self.batch_pass(x_train, y_train, loss, output_optimizer, mu=mu, reg=self.param_norm, args_reg=[1])
#        print(self.sparsity())
        self.sparsify_thres()
#        print(self.sparsity())
        """
            perform BFS
        """
        self.compute_hooks()
        self.register_hooks()
        """
        #  train subnetwork
        """

        #init book-keeping
        train_losses = []
        train_accs = []
        for i in range(n_epochs):
            self.batch_pass(x_train, y_train, loss, optimizer, reg=self.param_norm, args_reg=[2])

            #eval network's loss and acc
            train_l,_,train_acc = helper.evaluation(self, loss, x_train, y_train, 2)
            train_accs.append(train_acc)
            train_losses.append(train_l)

        helper.plot_curves([train_losses],'DEN','loss selec. retrain', filename="loss_task"+str(self.num_tasks))
        helper.plot_curves([train_accs],'DEN','accuracy selec. retrain', filename="acc_task"+str(self.num_tasks))

        self.unhook()
        return train_losses[-1]

    def dynamic_expansion(self, x_train, y_train, loss, retrain_loss, tau=0.02, n_epochs=10):  
        #TODO FIGURE OUT NB NEURON TO ADD
        nb_add_neuron = 30
        learning_rate = 0.1

        #if given loss isn't low enough, expand network
        if (retrain_loss > tau):

            #add new units
            for l in range(self.depth-1):
                self.add_neurons(l,nb_add_neuron)
            print(self)
            #train newly added neurons

            #first register hook for each layer
            for i,l in enumerate(self.layers):
                #define hook depending on considered layer
                if i == 0:
                    def my_hook(grad):
                        grad_clone = grad.clone()
                        grad_clone[:-nb_add_neuron,:] = 0
                        return grad_clone
                elif i == self.depth-1:
                    def my_hook(grad):
                        grad_clone = grad.clone()
                        grad_clone[:,:-nb_add_neuron] = 0
                        return grad_clone
                else: #hidden layers
                    def my_hook(grad):
                        grad_clone = grad.clone()
                        grad_clone[:-nb_add_neuron,:-nb_add_neuron] = 0
                        return grad_clone

                #register hook to weight variable
                l.weight.register_hook(my_hook)

            #train added neurons, layer per layer, with l1 norm for sparsity
            for l in self.layers:
                optimizer = t.optim.SGD(l.parameters(), lr=learning_rate)
                for i in range(n_epochs):
                       self.batch_pass(x_train, y_train, loss, optimizer)
            #remove useless units among the added ones
            for l in self.layers:
                pass
                #TODO REMOVE USELESS UNITS
        else:
            print("loss: " + str(retrain_loss) + ",low enough, dynamic_expansion not required")
        

    def duplicate(self, x_train, y_train, loss, optimizer, old_params_list, n_epochs=10, sigma=.002, lambd=.1):
        # Retrain network once again
        for i in range(n_epochs):
            self.batch_pass(x_train, y_train, loss, optimizer, mu=lambd, reg=self.drift, args_reg=[old_params_list])
        # Compute connection-wise distance
        for num_layer,layer in enumerate(self.layers):
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
        # Retrain
        for i in range(n_epochs):
            self.batch_pass(x_train, y_train, loss, optimizer, mu=lambd, reg=self.drift, args_reg=[old_params_list])
        pass



def add_output_dim(old_layer, n_neurons=1):
    """
    adds neurons to a layer
    """
    # WARNING Probably kills cuda
    input_dim, output_dim = old_layer.in_features, old_layer.out_features
    if old_layer.bias is not None:
        new_layer = nn.Linear(input_dim, output_dim + n_neurons)
        new_layer.bias[:-n_neurons].data = old_layer.bias.data
    else:
        new_layer = nn.Linear(input_dim, output_dim + n_neurons, bias=False)
    new_layer.weight[:-n_neurons].data = old_layer.weight.data
    return new_layer


def add_input_dim(old_layer, n_neurons=1):
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
    new_layer.weight[:,:-n_neurons].data = old_layer.weight.data
    return new_layer


def make_hook(hook):
#    print(hook.shape)
    def hooker(grad):
#        print(hook.shape, grad.shape)
        return grad * Variable(hook, requires_grad=False)
#    print("hooked)
    return hooker
