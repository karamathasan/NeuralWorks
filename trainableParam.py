import types
import numpy as np

class TrainableParameter():
    def __init__(self, paramName: str, value: np.ndarray, derivativeFunc = None):
        if (derivativeFunc != None):
            assert type(derivativeFunc) is types.LambdaType
        self.name = paramName
        self.derivativeFunc = derivativeFunc
        self.value = value

    def set(self, value: np.ndarray):
        self.value = value

    def get(self):
        return self.value

    def dZdTheta(self):
        '''
        let Z be equal to the function that theta, the current tunable paramter, is being used in for the supplied derivative
        this function will apply the derivative on the current value of theta
        '''
        return self.derivativeFunc(self)
    
    def __add__(self, other):
        return self.value + other

    def __radd__(self, other):
        return other + self.value

    def __iadd__(self, other):
        return TrainableParameter(self.name, self.value + other, self.derivative)

    def __sub__(self, other):
        return self.value - other

    def __rsub__(self, other):
        return other - self.value
    
    def __isub__(self, other):
        return TrainableParameter(self.name, self.value - other, self.derivative)

    # assumed to be scalar multiplication
    def __mul__(self, other):
        return self.value * other

    def __rmul__(self, other):
        return other * self.value
    
    def __imul__(self, other):
        return TrainableParameter(self.name, self.value * other, self.derivative)

    def __truediv__(self, other):
        return self.value / other

    def __rtruediv__(self, other):
        return other / self.value

    def __pow__(self, other):
        return self.value ** other

    def __rpow__(self, other):
        return other ** self.value

    def __neg__(self):
        return -self.value

    def __pos__(self):
        return +self.value

    def __abs__(self):
        return abs(self.value)

    def __str__(self):
        return f"TunableParameter(name={self.name}, value={self.value})"
    
    def dot(self, other):
        return self.value.dot(other)


class Weight(TrainableParameter):
    def __init__(self, paramName: str, neuron):
        self.neuron = neuron

        # derivativeFunc = lambda x
        value = self.initialize(neuron.numConnections)
        super().__init__(paramName, value)
        pass

    def initialize(self,n):
        randoms = np.array([])
        for i in range(n):
            randoms = np.append(randoms, np.random.uniform(-1,1))
    
        weights = randoms
        return weights * np.sqrt(1/n)

    def dZdTheta(self):
        return self.neuron.input
    
class Bias(TrainableParameter):
    def __init__(self, paramName: str, neuron):
        value = 0
        super().__init__(paramName,value)

    def dZdTheta(self):
        return 1
# class Gamma(TrainableParameter):

# class Beta(TrainableParameter):