import numpy as np
import types
import inspect

class LossFunction:
    def __init__(self, lossFunction, lossDerivative):
        '''
        Params:
            lossFunction: Lambda (y_true, y_pred) 
            lossDerivative: Lambda (y_true, y_pred)
        '''
        assert (type(lossFunction) == types.LambdaType and type(lossDerivative) == types.LambdaType), "FUNCTIONS ARE NOT BOTH OF TYPE LAMBDA!!!"
        assert (len(inspect.signature(lossFunction).parameters) == 2 and len(inspect.signature(lossDerivative).parameters) == 2), "FUNCTIONS HAVE INCORRECT NUMBER OF PARAMETERS!!!"
        self.func = lossFunction
        self.derivative = lossDerivative
        self.y_pred = 0
        self.y_true = 0
    
    def setInput(self,y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def evaluateAsArray(self,y_true, y_pred):
        '''
        returns the output of the loss function as vector
        each component of the vector refers to
        '''
        self.setInput(y_true, y_pred)
        return self.func(y_true, y_pred)
    
    def evaluateAsSum(self, y_true, y_pred):
        self.setInput(y_true, y_pred)
        vec = self.func(y_true, y_pred)
        return np.sum(vec)
    
    def evaluateDerivativeAsArray(self):
        '''
        evaluates the derivative with the current values of y_pred and y_true
        '''
        return self.derivative(self.y_pred, self.y_true)
    
    def evaluateDerivativeAsSum(self):
        '''
        evaluates the derivative with the current values of y_pred and y_true
        '''
        vec = self.derivative(self.y_pred, self.y_true)
        return np.sum(vec)
    
    def setLossFunc(self, lossFunction, lossDerivative):
        self.func = lossFunction    
        self.derivative = lossDerivative

class SquaredError(LossFunction):
    def __init__(self):
        super().__init__(lambda y_true, y_pred: (y_true-y_pred) ** 2,lambda y_true, y_pred : -2*(y_true-y_pred))
    
    def setLossFunc(self, lossFunction = None, lossDerivative = None):
        return NotImplementedError
    
class BinaryCrossEntropy(LossFunction):
    def __init__(self):
        lossFunction = lambda y_true, y_pred: (-1/len(y_true)) * (y_true * np.log(y_pred) + (-1/len(y_true)) *(1-y_true)*np.log(1-y_pred))
        lossDerivative = lambda y_true, y_pred: (-1/len(y_true)) * (y_true/y_pred) + (1/len(y_true)) * (1-y_true/1-y_pred)        
        super().__init__(lossFunction, lossDerivative)

    def setLossFunc(self, lossFunction = None, lossDerivative = None):
        return NotImplementedError   
    
# class CrossEntropy(LossFunction):
#     def __init__(self):
#         super().__init__(lossFunction, lossDerivative)

#     def setLossFunc(self, lossFunction = None, lossDerivative = None):
#         return NotImplementedError