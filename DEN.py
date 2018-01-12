import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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
            return x


    def param_norm(self, p=2):
        norm = 0
        for l in list(self.parameters()):
            norm += l.norm(p)
        return norm

    def sparsify(self, old_params_list):
        for i, l in enumerate(self.parameters()):
            mask = (l*old_params_list[i]) > 0
            l.data *= mask.data.float()

    def sparsity(self):
        num = 0
        denom = 0
        for i, l in enumerate(self.parameters()):
            num += (l != 0).sum().data[0]
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
        if l > (self.depth - 2):
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




#            self.hook_handles.append(l.bias.register_hook(lambda grad : print(grad.shape, book.shape)))#, grad * Variable(self.b_hooks[i], requires_grad=False)]))
#            self.hook_handles.append(l.weight.register_hook(lambda grad : grad * Variable(self.w_hooks[i], requires_grad=False)))

    def unhook(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def batch_pass(self, x_train, y_train, loss, optim, mu=0.1, p=2, batch_size=32, cuda=False):
        #print(list(self.parameters()))
        train_size = x_train.shape[0]
        if(self.num_tasks != 1):
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
            l = mu* self.param_norm()
            optim.zero_grad()
            l.backward()
            optim.step()

    def selective_retrain(self, x_train, y_train, loss, optimizer, n_epochs=1):
        """
        Retrain output layer
        """
        #Solving for output layer
        output_optimizer = t.optim.SGD(self.layers[-1].parameters(), lr=0.1, weight_decay=0)
        # train it
        for i in range(n_epochs):
            self.batch_pass(x_train, y_train, loss, output_optimizer, p=1)
        """
            perform BFS
        """
        self.compute_hooks()
        self.register_hooks()

        # train subnetwork

        """
        for i in range(n_epochs):
            self.batch_pass(x_train, y_train, loss, optimizer, p=2)
        """

        self.unhook()

    def dynamic_expansion(self, loss, tau=0.02, n_epochs=10):
    	#TODO FIGURE OUT NB NEURON TO ADD
    	nb_add_neuron = 30
        # Perform selective retraining and compute loss, or get it as param
        if (loss > tau):
            #add new units
            for l in range(depth-1):
            	self.add_neurons(l,nb_add_neuron)

            #retrain network

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
					    grad_clone[0,:-nb_add_neuron] = 0
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
            	optimizer = optim.SGD([l.weight,l.bias], lr=learning_rate, weight_decay=0)
	        	for i in range(n_epochs):
	            	self.batch_pass(x_train, y_train, loss, optimizer, p=1)

	        #remove useless units among the added ones
	        for l in self.layers:
	        	pass
	        	#TODO REMOVE USELESS UNITS


        pass

    def duplicate(self, sigma=.002):
        # Retrain network once again ?
        # Compute connection-wise distance
        # If distance > sigma, add old neuron to network
        # Retrain
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
