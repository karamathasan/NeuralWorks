import numpy as np
import neuron as n
import random 
'''
the layer contains neurons. each neuron has an array/vector of weights associated with its output, another array/ vector of biases and an activation function
the length of these vectors should be equal to the size of the next layer
'''
class Layer():
    # def __init__(self, neurons):
    #     self.neurons = []
    #     for i in range(neurons):
    #         self.neurons.append(n.Neuron.createRandom())
    def __init__(self, size, connections):
        self.neurons = []
        self.connections = connections
        for i in range(size):
            # generate 'size' many neurons with random weights and biases that output to 'connections' many inputs in the preceding layer 
            self.neurons.append(n.Neuron(connections))

    def getSize(self):
        return len(self.neurons)
    
    def resetConnections(self, connections):
        self.connections = connections
        for i in range(len(self.neurons)):
            self.neurons[i] = n.Neuron(connections)

    def evaluate(self, input, nextLayer = None):
        assert (len(input) == len(self.neurons))
        if (nextLayer == None):
            return input

        output = np.array([])
        for j in range(len(nextLayer.neurons)):
            sum = self.getWeightedSum(input,j)
            sum += nextLayer.neurons[j].bias
            output = np.append(output, nextLayer.neurons[j].activate(sum))
        return output

            
    '''
    returns a vector of the i-th weights of all neurons
    '''
    # def getLayerWeight(self,i):
    #     weights = np.array()
    #     for neuron in self.neurons:
    #         weights.append(neuron.weights[i])

    def getWeightedSum(self, input, outputIndex):
        assert (len(input) == len(self.neurons))
        weightedSum = 0
        for i in range(len(input)):
            weightedSum += self.neurons[i].weights[outputIndex] * input[i]
        return weightedSum
    
    def equals(self, other):
        result = True
        if (len(self.neurons) != len(other.neurons)):
            return False
        for i in range(len(self.neurons)):
            result = result and (self.neurons[i].equals(other.neurons[i]))
        result = result and (self.connections == other.connections)
        return result