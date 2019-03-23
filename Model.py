import numpy as np
import math
from data_handler import encode, decode

class Model():

    def __init__(self,  H, nLayers=1, V=153, isTrain=True):
        self.nLayers = nLayers
        self.H = H
        self.V = V
        self.D = int(math.log(V,2))+1
        self.layers = []
        self.isTrain = isTrain

    def add_layer(self, layer):
        self.layers.append(layer)
 
    def __call__(self, x):
        return self.forward(x)

    def forward(self, inp):
        for layer in self.Layers:
            inp = layer(inp)
        self.output = inp
        return inp
