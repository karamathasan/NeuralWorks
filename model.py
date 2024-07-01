import math
import numpy as np
import pandas as pd

import layer as l
from data_helper import shuffle
class Model():
    # def __init__(self, NumberOfInputs, Layers):
    #     # needs more parameters
    #     pass

    '''
    inputSize refers to the length of the vector associated with the input layer. likewise for the outputSize
    the output layer SHOULD NOT use a layer object, and instead should use a vector
    '''
    def __init__(self, inputSize = 1, outputSize = 1, activationFunc = None):
        self.inputLayer = l.Layer(inputSize,1, activationFunc)
        self.hiddenLayers = []
        self.outputLayer = l.Layer(outputSize,0, activationFunc)
        # print(self.outputLayer.bias)

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

    def train(self, predictor_data, effector_data, train_method = "mini-batch"):
        assert(predictor_data.shape[0] == effector_data.shape[0])
        def backprop(layer: l.Layer, ssr: float):
            pass
        # ssr = 0
        # for row in range(predictor_data.shape[0]):
        #     prediction = self.predict(predictor_data[row])
        #     residual = effector_data[row] - prediction
        #     residual *= residual
        #     ssr += residual
        # return ssr
        if train_method == "mini-batch":
            shuffle(predictor_data)
            num_batches = 10
            batch_size = int(len(predictor_data)/num_batches)
            ssr = 0 
            for i in range(len(predictor_data)):
                if (i % batch_size == 0):
                    # back propogate and update parameters
                    ssr == 0
                prediction = self.predict(predictor_data[i])
                residual = effector_data[i] - prediction
                residual *= residual
                ssr += residual
                
            return ssr
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

    def getNeurons(self):
        neurons = []
        neurons.append(self.inputLayer.getNeurons())
        for hiddenLayer in self.hiddenLayers:
            neurons.append(hiddenLayer.getNeurons())
        neurons.append(self.outputLayer.getNeurons())
        return neurons