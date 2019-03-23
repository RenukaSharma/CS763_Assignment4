import numpy as np

class Criterion():

    def softmax(self, x):
        # dim(x) = [input_dim, batch_size]
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def forward(self, input, target):
        self.x = self.softmax(input.copy())
        loss = -np.mean(y.copy()*np.log(self.x))
        
