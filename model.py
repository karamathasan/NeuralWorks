import numpy as np
import pandas as pd

import layer as l
from data_helper import shuffle
from backprop_helper import backpropagate

class Model():
    # def __init__(self, NumberOfInputs, Layers):
    #     # needs more parameters
    #     pass

    '''
    inputSize refers to the length of the vector associated with the input layer. likewise for the outputSize
    the output layer SHOULD NOT use a layer object, and instead should use a vector
    '''
    def __init__(self, inputSize = 1, outputSize = 1, activationFunc = None, learningRate = 0.01):
        self.inputSize = inputSize
        self.hiddenLayers = []
        self.learningRate = learningRate
        if (activationFunc != None):
            self.defaultActivation = activationFunc
        else:
            self.defaultActivation = "relu"
        self.outputLayer = l.Layer(outputSize, inputSize, activationFunc)

    def predict(self, input):
        allLayers = self.getLayers()
        output = input
        for i in range(len(allLayers)):
            # output = self.hiddenLayers[0].evaluate(input)
            currentLayer = allLayers[i]
            output = currentLayer.evaluate(output)
        return output

    def train(self, predictor_data: pd.DataFrame, effector_data: pd.DataFrame, train_method = "iterative"):
        assert(predictor_data.shape[0] == effector_data.shape[0])      
        if train_method == "mini-batch":
            shuffle(predictor_data)
            num_batches = 10
            batch_size = int(len(predictor_data)/num_batches)

            residuals = []
            for i in range(len(predictor_data)):
                row = predictor_data.iloc[i].to_numpy()
                prediction = self.predict(row)
                residuals.append(effector_data.iloc[i].to_numpy() - prediction)

                if (i % batch_size == 0):
                    backpropagate(self,residuals)
                    residuals = []            
                    # return
        elif train_method == "iterative":
            residuals = []
            for i in range(len(predictor_data)):
                row = predictor_data.iloc[i].to_numpy()
                prediction = self.predict(row)
                residuals.append(effector_data.iloc[i].to_numpy() - prediction)
                backpropagate(self,residuals)
                residuals = []  
        print("training complete!")

    def test(self, predictor_data: pd.DataFrame, effector_data: pd.DataFrame):
        ssr = 0 # sum of squared residuals
        sape = 0 # sum of absolute percent error
        for i in range(len(predictor_data)):
            prediction = self.predict(predictor_data.iloc[i].to_numpy())
            truth = effector_data.iloc[i].to_numpy()
            residual = truth - prediction
            sape += np.abs(residual) / (truth + 0.00001)
            ssr += residual * residual
        print("testing complete")
        mse = ssr / len(predictor_data)
        print(f"mean squared error: {mse}")
        mape = sape / len(predictor_data)
        print(f"mean absolute percent error: {mape}")

    '''
    layerSize should be a integer referring to the size of the layer 
    '''
    def addHiddenLayer(self, layerSize):
        numHiddenLayers = len(self.hiddenLayers)
        if numHiddenLayers == 0:
            newLayer = l.Layer(layerSize, self.getInputSize(), self.defaultActivation)
            self.hiddenLayers.append(newLayer)
            self.outputLayer.resetConnections(layerSize, self.defaultActivation)
        else:
            prevLayer = self.hiddenLayers[numHiddenLayers-1]
            prevLayerSize = prevLayer.getSize()
            newLayer = l.Layer(layerSize, prevLayerSize, self.defaultActivation)
            self.hiddenLayers.append(newLayer)
            self.outputLayer.resetConnections(layerSize, self.defaultActivation)
            # self.outputLayer.resetConnections(newLayer.getSize(), self.defaultActivation)


    def getInputSize(self):
        return self.inputSize
    
    def setInputSize(self, size):
        self.inputSize = size
        self.hiddenLayers[0].resetConnections(size, self.defaultActivation)

    def getOutputSize(self):
        return self.outputLayer.getSize()
    
    def setOutputSize(self, size):
        lastHiddenLayer = self.hiddenLayers[len(self.hiddenLayers)-1]
        self.outputLayer = l.Layer(size, lastHiddenLayer.getSize() ,self.defaultActivation)

    def modelShape(self):
        print(f"input: {self.getInputSize()}")
        for layerNum in range(len(self.hiddenLayers)):
            print(f"hidden layer {layerNum}: {self.hiddenLayers[layerNum].getSize()}") 
        print(f"output: {self.getOutputSize()}") 

    def neuronCount(self):
        count = 0
        # count += len(self.inputLayer.getNeurons())
        for layer in self.hiddenLayers:
            count += len(layer.getNeurons())
        count += len(self.outputLayer.getNeurons())

    def getNeurons(self):
        neurons = []
        # neurons.append(self.inputLayer.getNeurons())
        for hiddenLayer in self.hiddenLayers:
            neurons.append(hiddenLayer.getNeurons())
        neurons.append(self.outputLayer.getNeurons())
        return neurons
    
    def getLayerByIndex(self,index):
        if (index < len(self.hiddenLayers)):
            return self.hiddenLayers[index]
        elif(index == len(self.hiddenLayers)):
            return self.outputLayer
        return None
    
    def getLayers(self):
        allLayers = []
        allLayers.extend(self.hiddenLayers)
        allLayers.append(self.outputLayer)
        return allLayers