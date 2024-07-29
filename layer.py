import numpy as np
import neuron as n

class BaseLayer():
    '''
    base class for all kinds of layers. 

    a "layer" is a computational object that has inputs x and activation of y where x and y are both vectors and y is a function of x
    ex. in a dense layer, y = sigma(wx+b)
    ex. in a batch layer y = gama xHat + beta where xHat is the batch normalized value for the input, x
    '''
    def __init__(self, size, inputs, model):
        pass
    
    def evaluate(self):
        pass

    def resetinputs(self):
        pass

    def getParams(self):
        pass

    def dOutdIn(self, inIndex, outIndex):
        pass
    
class Layer():
    '''
    the typical fully connected layer of neurons. 
    Args:
        size: the number of neurons in this layer
        inputs: the number of inputs to this layer. Should be equal to the size of the previous layer
        model: the model associated with the layer
        activationFunc: the activation function of each neuron in this layer
    '''
    def __init__(self, size, inputs, model, activationFunc = None):
        self.neurons = []
        self.inputs = inputs
        self.model = model
        self.defaultActivation = activationFunc
        for i in range(size):
            self.neurons.append(n.Neuron(inputs, activationFunc))

    def getSize(self):
        return len(self.neurons)
    
    def setActivation(self, activationFunc):
        self.inputs 
        for i in range(len(self.neurons)):
            if (activationFunc != None):
                self.neurons[i] = n.Neuron(self.inputs, activationFunc)

    def resetInputs(self, inputs, activationFunc = None):
        self.inputs = inputs
            
        for i in range(len(self.neurons)):
            if (activationFunc != None):
                self.neurons[i] = n.Neuron(inputs, activationFunc)
            else:
                self.neurons[i] = n.Neuron(inputs, self.defaultActivation)

    def evaluate(self, input: np.array):
        assert(input.shape[0] == self.inputs)
        output = np.zeros(len(self.neurons))
        for i in range(len(self.neurons)):
            output[i] = self.neurons[i].evaluate(input)
        return output
    
    def equals(self, other):
        result = True
        if (len(self.neurons) != len(other.neurons)):
            return False
        for i in range(len(self.neurons)):
            result = result and (self.neurons[i].equals(other.neurons[i]))
        result = result and (self.inputs == other.inputs)
        return result
    
    def getNeurons(self):
        return self.neurons

    def getLayerActivation(self):
        layerActivation = np.array([])
        for n in self.neurons:
            layerActivation = np.append(layerActivation, n.activation)
        return layerActivation
    
    def getLayerActivationDerivative(self):
        activationDerivative = np.array([])
        for neuron in self.neurons:
            activationDerivative = np.append(activationDerivative, neuron.activationDerivative())
        return activationDerivative

class BatchNorm(BaseLayer):
    def __init__(self, size, model):
        '''
        creates a Batch Normalization layer
        The inputs to this layer are 
        '''
        self.size = size
        self.inputs = size
        self.model = model
        
        self.gamma = 0
        self.beta = 0
        self.batchMean = 0
        self.batchVariance = 0
        self.rollingMean = 0
        self.rollingVariance = 0