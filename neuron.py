import math
import numpy as np
import random

class Neuron():
    # def __init__(self, weight, bias, activation):
    #     self.weights = weight
    #     self.biases = bias
    #     self.activation = activation

    def __init__(self, connections, activationFunc = None):
        self.weights = self.createRandoms( -1,1,connections)
        # self.baises = self.createRandoms(self,-1,1,connections)
        self.bias = random.uniform(-1,1)
        if (activationFunc != None):
            self.activationFunction = activationFunc
        else:
            self.activationFunction = "relu"
    
    def createRandoms(self, min, max, quantity):
        # randoms = []
        randoms = np.array([])
        for i in range(quantity):
            randoms = np.append(randoms, random.uniform(min,max))
        return randoms

    def evaluate(self, input, index):
        rawVal = self.weights[index] * input + self.bias
        # rawVal = np.min(rawVal,1)
        if self.activationFunction == "lin":
            return rawVal
        elif self.activationFunction == "tanh":
            return math.tanh(rawVal)
        elif self.activationFunction == "relu":
            return np.max(rawVal,0)
        elif self.activationFunction == "sigmoid":
            return 1 / (1 + np.exp(-rawVal))
        else:
            print("NO ACTIVATION FUNCTION FOUND!!")
            return
        
    def setActivation(self, funcName):
        self.activationFunction = funcName
        
    def activate(self, input):
        if self.activationFunction == "lin":
            return input
        elif self.activationFunction == "tanh":
            return math.tanh(input)
        elif self.activationFunction == "relu":
            return np.max(input,0)
        elif self.activationFunction == "sigmoid":
            return 1 / (1 + np.exp(-input))
        else:
            print(f"NO AVAILABLE ACTIVATION FUNCTION: {input}")
            return
        
    def equals(self, other):
        result = True
        if (len(self.weights) != len(other.weights)):
            return False
        for i in range(len(self.weights)):
            result = result and (self.weights[i] == other.weights[i])
        result = result and (self.bias == other.bias)
        result = result and (self.activationFunction == other.activationFunction)
        return result
    