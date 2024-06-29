import math
import numpy as np

class Neuron():
    def __init__(self, weight, bias, activation):
        self.weight = weight
        self.bias = bias
        self.activation = activation
        
    def evaluate(self, input):
        rawVal = self.weight * input + self.bias
        if self.activation == "lin":
            return rawVal
        elif self.activation == "relu":
            return np.max(rawVal,0)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-rawVal))
    