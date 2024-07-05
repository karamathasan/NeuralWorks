import numpy as np
import neuron as n

'''
the layer contains neurons. each neuron has an array/vector of weights associated with its output, another array/ vector of biases and an activation function
the length of these vectors should be equal to the size of the next layer
'''
class Layer():
    # the direction of connections is backward 
    def __init__(self, size, connections, activationFunc = None):
        self.neurons = []
        self.connections = connections
        for i in range(size):
            # generate 'size' many neurons with random weights and biases that output to 'connections' many inputs in the preceding layer 
            self.neurons.append(n.Neuron(connections, activationFunc))

    def getSize(self):
        return len(self.neurons)
    
    def resetConnections(self, connections, activationFunc):
        self.connections = connections
            
        for i in range(len(self.neurons)):
            if (activationFunc != None):
                self.neurons[i] = n.Neuron(connections, activationFunc)
            else:
                self.neurons[i] = n.Neuron(connections, "relu")

    def evaluate(self, input: np.array):
        assert(input.shape[0] == self.connections)
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
        result = result and (self.connections == other.connections)
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