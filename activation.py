import numpy as np
import math
import types
import inspect

class ActivationFunction():
    def __init__(self, activationFunc, activationDerivative):
        assert (type(activationFunc) == types.LambdaType and type(activationDerivative) == types.LambdaType), "FUNCTIONS ARE NOT BOTH OF TYPE LAMBDA!!!"
        assert (len(inspect.signature(activationFunc).parameters) == 1 and len(inspect.signature(activationDerivative).parameters) == 1), "FUNCTIONS HAVE INCORRECT NUMBER OF PARAMETERS!!!"
        self.func = activationFunc
        self.derivative = activationDerivative
        self.input = 0
    
    def setInput(self,input):
        self.input = input
    
    def evaluate(self, input):
        self.input = input
        return self.func(input)
    
    def evaluateDerivative(self):
        return self.derivative(self.input)

class Relu(ActivationFunction):
    def __init__(self):
        activationFunc = lambda x : np.maximum(0,x)
        activationDerivative = lambda x :  0 if x < 0 else 1 
        super().__init__(activationFunc, activationDerivative)
    
class Sigmoid(ActivationFunction):
    def __init__(self):
        # activationFunc = lambda x : (1/(1+np.exp(-x)))
        activationFunc = lambda x : (1/(1+np.exp(-x))) if x > 0 else (np.exp(x)/(1+np.exp(x)))

        activationDerivative = lambda x :  activationFunc(x) * (1-activationFunc(x))
        super().__init__(activationFunc, activationDerivative)