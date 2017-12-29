class DEN(nn.module):
    def __init__(self, sizes):
        super(DEN, self).__init__()
        self.depth = len(sizes)
        self.sizes = sizes
        self.num_tasks = 0
        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]), nn.ReLU() for i in range(selfdepth-1)])
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
            for i, l in enumerate(self.layers):
                x = l(x)
            x = self.softmax(x)
            return x


def selective_retrain(DEN):
    """
    Retrain output layer
    """
    layer = depth-1
    #Solving for output layer
    optimizer = optim.SGD(self.layers[-1].parameters(), lr=learning_rate, weight_decay=0)
    #TODO train it
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
    #TODO train subnetwork
    optimizer = optim.SGD(params, lr=learning_rate, weight_decay=0)
