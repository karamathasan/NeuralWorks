import math
import numpy as np
import random

class Neuron():
    def __init__(self, numConnections, activationFunc = None):
        self.weights = self.createRandoms( -1,1,numConnections)
        self.numConnections = numConnections
        self.bias = random.uniform(-1,1)
        # self.bias = 0
        self.input = None
        self.activation = 0
        # self.connectedNodes = []

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

    def evaluate(self, input: np.array):
        assert (input.shape[0] == self.numConnections)
        self.input = input
        rawVal = self.weights.dot(input) + self.bias
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
        
    # def activationDerivative(self, input):
    #     if self.activationFunction == "lin":
    #         return 1
    #     elif self.activationFunction == "tanh":
    #         return 1/math.cosh(input) * math.cosh(input)
    #     elif self.activationFunction == "relu":
    #         if (input <= 0):
    #             return 0
    #         else:
    #             return 1
    #     elif self.activationFunction == "sigmoid":
    #         return (1 / (1 + np.exp(-input))) * (1 - (1 / (1 + np.exp(-input))))
    #     else:
    #         print(f"NO AVAILABLE ACTIVATION FUNCTION: {input}")
    #         return

    def activationFunctionDerivative(self):
        if self.activationFunction == "lin":
            return 1
        elif self.activationFunction == "tanh":
            return 1/math.cosh(self.input) * math.cosh(self.input)
        elif self.activationFunction == "relu":
            if (self.input <= 0):
                return 0
            else:
                return 1
        elif self.activationFunction == "sigmoid":
            return (1 / (1 + np.exp(-self.input))) * (1 - (1 / (1 + np.exp(-self.input))))
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
    
    def __str__(self):
        return f"weights: {self.weights} \nbias: {self.bias} \nnum connections: {self.numConnections}" 