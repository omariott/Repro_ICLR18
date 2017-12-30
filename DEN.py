import torch as t
import torch.nn as nn

class DEN(nn.Module):
    def __init__(self, sizes):
        super(DEN, self).__init__()
        self.depth = len(sizes)
        self.sizes = sizes
        self.num_tasks = 1
        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) if odd-1 else nn.ReLU() for i in range(self.depth-1) for odd in range(2)] + [nn.Linear(sizes[-1], 1)])

    def forward(self, x):
            for i, l in enumerate(self.layers):
                if i<self.depth*2-1:
                    x = l(x)
            return x



    def param_norm(self, p=2):
        norm = 0
        for i, l in enumerate(self.layers):
            norm += l.weight.norm(p)
        return norm

    def add_task(self):
        # WARNING Probably kills cuda
        self.num_tasks += 1
        #add output neuron
        newout = nn.Linear(self.sizes[-1], self.num_tasks)
        oldout = self.layers[-1]
        newout.weight[:-1] = oldout.weight
        if oldout.bias:
            newout.bias[:-1] = oldout.bias
        self.layers[-1] = newout


    def batch_pass(self, x_train, y_train, loss, optim, mu=0.1, p=2, batch_size=32, cuda=False, first_task=False):
        if(first_task):
            for i in range(train_size // batch_size):
                #load batch
                indsBatch = range(i * batch_size, (i+1) * batch_size)
                x = Variable(x_train[indsBatch, :], requires_grad=False)
                y = Variable(y_train[indsBatch], requires_grad=False)
                if cuda: x,y = x.cuda(), y.cuda()

                #forward
                y_til = self.forward(x)
                #loss and backward
                l = loss(y_til,y) + mu * self.param_norm(p)
                optim.zero_grad()
                l.backward()
                optim.step()


    def selective_retrain(self, x_train, y_train, loss, n_epochs=10):
        """
        Retrain output layer
        """
        layer = depth-1
        #Solving for output layer
        optimizer = optim.SGD(self.layers[-1].parameters(), lr=learning_rate, weight_decay=0)
        # train it
        for i in range(n_epochs):
            self.batch_pass(x_train, y_train, loss, optimizer, p=1)
        """
            perform BFS
        """
        #mask of selected neurons for output layer, we only get the last one corresponding to the new tasks
        mask = np.zeros(self.num_tasks)
        mask[-1] = 1
        #list of masks for each layer
        selected_neurons = [[mask]]
        #list of parameters to retrain
        params = []
        while(layer > 0):
            layer_size = self.sizes[layer]
            new_mask = np.zeros(layer_size)
            old_mask = selected_neurons[-1]
            connections = self.layers[layer].weight
            for i in range(layer_size):
                if(connections[i].dot(old_mask).sum() > 0):
                    new_mask[i] = 1
            params.append(connections[new_mask])
            selected_neurons.append(new_mask)
            layer -= 1
        # train subnetwork
        optimizer = optim.SGD(params, lr=learning_rate, weight_decay=0)
        for i in range(n_epochs):
            self.batch_pass(x_train, y_train, loss, optimizer, p=2)

    def dynamic_expansion(self, loss, tau=0.02):
        # Perform selective retraining and compute loss, or get it as param
        if (loss > tau):
            #TODO Add units
            #TODO retrain
            pass
        #TODO remove neurons with no connections
        pass

    def duplicate(self, sigma=.002):
        # Retrain network once again ?
        # Compute connection-wise distance
        # If distance > sigma, add old neuron to network
        # Retrain
        pass
