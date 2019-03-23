import numpy as np
from data_handler import encode, decode

def sigmoid(x):
    return  1/(1+np.exp(-x))

class RNN():
    # size of input for each layer is 8

    def __init__(self, input_dim=8, hidden_dim=64, weights=None, initialisation='same'):
        
        # initialisation of weights
        if initialisation=='gaussian':
            self.initialise()
        elif initialisation=='same':
            self.Whh, self.Wxh, self.Why, self.bhh, self.bxh, self.bhy = weights

    def __call__(self, x):
        return self.forward(x)

    def initialise(self):
        self.Whh = np.random.normal(loc=0.0,scale=0.1,size=self.hidden_dim*self.hidden_dim).reshape([self.hidden_dim,self.hidden_dim])
        self.Wxh = np.random.normal(loc=0.0,scale=0.1,size=self.hidden_dim*self.input_dim).reshape([self.hidden_dim,self.input_dim])
        self.Why = np.random.normal(loc=0.0,scale=0.1,size=self.hidden_dim).reshape([1,self.hidden_dim])
        self.bh = np.zeros([self.hidden_dim,1])
        self.by = np.zeros([1,1])
    
    def forward(self, inp):
        self.input = inp.copy()
        # initialise the list of hidden nodes with h0 - all zeros
        # input dimensions = [input_dim, seq_length, batch_size]
        input_dim, seq_length, batch_size = np.shape(inp)
        hidden_list = [np.zeros([self.hidden_dim,batch_size])]
        for t in range(seq_length):
            x = inp(:,t)
            hidden = np.tanh(np.matmul(self.Whh,hidden_list[-1])+self.bh + np.matmul(self.Wxh,x))
            hidden_list.append(hidden)
        self.output = np.tanh(np.matmul(self.Why,hidden_list[-1]))+self.by
    
    def backward(self):
        return 
