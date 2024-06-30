import math
import numpy as np

import layer as l
class Model():
    # def __init__(self, NumberOfInputs, Layers):
    #     # needs more parameters
    #     pass

    '''
    inputSize refers to the length of the vector associated with the input layer. likewise for the outputSize
    the output layer SHOULD NOT use a layer object, and instead should use a vector
    '''
    def __init__(self, inputSize = 1, outputSize = 1):
        self.inputLayer = l.Layer(inputSize,1)
        self.hiddenLayers = []
        self.outputLayer = l.Layer(outputSize,0)
        # print(self.outputLayer.bias)

    def fit(self):
        pass

    def weightedSum(self):
        pass
    def predict(self, input):
        assert(len(input) == self.getInputSize())
        allLayers = np.append(self.inputLayer, self.hiddenLayers)
        allLayers = np.append(allLayers, self.outputLayer)
        
        output = input
        for i in range(len(allLayers)):
            currentLayer = allLayers[i]
            if (currentLayer.equals(self.outputLayer)):
                return output
            nextLayer = allLayers[i + 1]
            output = currentLayer.evaluate(output, nextLayer)
        return output
        # out = self.inputLayer.evaluate(input, self.hiddenLayers[0])
        # for layer in self.hiddenLayers:
        #     out = layer.evaluate(out, )
        # # print("output Layer began")
        # out = self.outputLayer.evaluate(out)
        # print(out)

    def train(self):
        pass

    def accuracy(self):
        pass

    '''
    layerSize should be a integer referring to the size of the layer 
    '''
    def addHiddenLayer(self, layerSize):
        newLayer = l.Layer(layerSize, self.getOutputSize())
        self.hiddenLayers.append(newLayer)
        if len(self.hiddenLayers) == 1:
            self.inputLayer = l.Layer(self.getInputSize(), layerSize)
        else:
            self.hiddenLayers[len(self.hiddenLayers)-2].resetConnections(layerSize)

    def getInputSize(self):
        return self.inputLayer.getSize()
    
    def setInputSize(self, size):
        self.inputLayer = l.Layer(size)

    def getOutputSize(self):
        return self.outputLayer.getSize()
    
    def setOutputSize(self, size):
        self.outputLayer = l.Layer(size)

    def modelShape(self):
        print(f"input: {self.getInputSize()}")
        for layerNum in range(len(self.hiddenLayers)):
            print(f"hidden layer {layerNum}: {self.hiddenLayers[layerNum].getSize()}") 
        print(f"output: {self.getOutputSize()}") 
