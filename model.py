import numpy as np
import pandas as pd

import loss 
import activation
import layer 
from data_helper import shuffle
from backprop_helper import backpropagate

class Model():
    '''
    inputSize refers to the length of the vector associated with the input layer. likewise for the outputSize
    the output layer SHOULD NOT use a layer object, and instead should use a vector
    '''
    def __init__(self, inputSize = 1, outputSize = 1, activationFunc = None, lossFunc = None, learningRate = 0.01):
        self.inputSize = inputSize
        self.hiddenLayers = []
        self.learningRate = learningRate
        
        if (activationFunc != None):
            assert isinstance(activationFunc, activation.ActivationFunction) or isinstance(lossFunc, str), "LOSS FUNCTION PARAMETER IS NOT A STRING OR OF TYPE ACTIVATIONFUCTION"
            self.defaultActivation = activationFunc
        else:
            self.defaultActivation = activation.Relu()

        if (lossFunc != None):
            assert isinstance(lossFunc, loss.LossFunction) or isinstance(lossFunc, str), "LOSS FUNCTION PARAMETER IS NOT A STRING OR OF TYPE LOSSFUCTION"
            self.lossFunc = lossFunc
        else:
            self.lossFunc = loss.SquaredError()
        self.outputLayer = layer.Layer(outputSize, inputSize, activationFunc)

    def predict(self, input):
        allLayers = self.getLayers()
        output = input
        for i in range(len(allLayers)):
            currentLayer = allLayers[i]
            output = currentLayer.evaluate(output)
        return output

    def train(self, predictor_data: pd.DataFrame, effector_data: pd.DataFrame, train_method = "sgd", epochs = 1):
        assert(predictor_data.shape[0] == effector_data.shape[0]) 
        for j in range(epochs):    

            if train_method == "mini-batch":
                shuffle(predictor_data)
                # num_batches = 10
                # batch_size = int(len(predictor_data)/num_batches)
                # num_batches = int(len(predictor_data)/ batchSize)
                batchSize = 2

                residuals = []
                for i in range(len(predictor_data)):
                    row = predictor_data.iloc[i].to_numpy()
                    prediction = self.predict(row)
                    residuals.append(effector_data.iloc[i].to_numpy() - prediction)

                    if (i % batchSize == 0):
                        backpropagate(self,residuals)
                        residuals = []            
                        
            elif train_method == "sgd":
                for i in range(len(predictor_data)):
                    row = predictor_data.iloc[i].to_numpy()
                    prediction = self.predict(row)
                    print(f"iteration: {j,i}")
                    print(f"    prediction: {prediction}")
                    print(f"    true: {effector_data.iloc[i].to_numpy()}")
                    # print(f"    percent error: {np.abs(effector_data.iloc[i].to_numpy() - prediction)/effector_data.iloc[i].to_numpy() * 100}%")
                    residuals = self.calculateResidualsArray(effector_data.iloc[i].to_numpy(), prediction)
                    backpropagate(self,residuals)
        print("training complete!")

    def calculateResidualsArray(self,y_true,y_pred):
        return self.lossFunc.evaluateAsArray(y_true,y_pred)

    def test(self, predictor_data: pd.DataFrame, effector_data: pd.DataFrame):
        ssr = 0 # sum of squared residuals
        sr = 0 # sum of residuals
        ape = 0
        for i in range(len(predictor_data)):
            prediction = self.predict(predictor_data.iloc[i].to_numpy())
            truth = effector_data.iloc[i].to_numpy()
            residual = truth - prediction
            ssr += residual * residual
            sr += np.abs(residual)
            ape += 100 * np.abs(residual)/truth
        mse = ssr / len(predictor_data)
        print(f"mean squared error: {mse}")
        mr =  sr / len(predictor_data)
        print(f"mean error: {mr}")
        mape = ape / len(predictor_data)
        print(f"mean absolute percent error: {mape}")

    def addHiddenLayer(self, layerSize):
        numHiddenLayers = len(self.hiddenLayers)
        if numHiddenLayers == 0:
            newLayer = layer.Layer(layerSize, self.getInputSize(), self.defaultActivation)
            self.hiddenLayers.append(newLayer)
            self.outputLayer.resetConnections(layerSize, self.defaultActivation)
        else:
            prevLayer = self.hiddenLayers[numHiddenLayers-1]
            prevLayerSize = prevLayer.getSize()
            newLayer = layer.Layer(layerSize, prevLayerSize, self.defaultActivation)
            self.hiddenLayers.append(newLayer)
            self.outputLayer.resetConnections(layerSize, self.defaultActivation)

    def getInputSize(self):
        return self.inputSize
    
    def setInputSize(self, size):
        self.inputSize = size
        self.hiddenLayers[0].resetConnections(size, self.defaultActivation)

    def getOutputSize(self):
        return self.outputLayer.getSize()
    
    def setOutputSize(self, size):
        lastHiddenLayer = self.hiddenLayers[len(self.hiddenLayers)-1]
        self.outputLayer = layer.Layer(size, lastHiddenLayer.getSize() ,self.defaultActivation)

    def modelShape(self):
        print(f"input: {self.getInputSize()}")
        for layerNum in range(len(self.hiddenLayers)):
            print(f"hidden layer {layerNum}: {self.hiddenLayers[layerNum].getSize()}") 
        print(f"output: {self.getOutputSize()}") 

    def neuronCount(self):
        count = 0
        for layer in self.hiddenLayers:
            count += len(layer.getNeurons())
        count += len(self.outputLayer.getNeurons())

    def getNeurons(self):
        neurons = []
        for hiddenLayer in self.hiddenLayers:
            neurons.append(hiddenLayer.getNeurons())
        neurons.append(self.outputLayer.getNeurons())
        return neurons
    
    def getLayerByIndex(self,index):
        if (index < len(self.hiddenLayers)):
            return self.hiddenLayers[index]
        elif(index == len(self.hiddenLayers)):
            return self.outputLayer
        print(f"LAYER NOT FOUND!! index requested: {index}")
        return None
    
    def getLayers(self):
        '''
        returns layers as an array
        '''
        allLayers = []
        allLayers.extend(self.hiddenLayers)
        allLayers.append(self.outputLayer)
        return allLayers
    
    def getParams(self, printParams = False):
        modelParams = []
        for j in range(len(self.getLayers())):
            modelParams.append([])
            for i in range(len(self.getLayers()[j].neurons)):
                if (printParams is True):
                    print(f"layer: {j}, neuron: {i}")
                    print(f"    weights: {self.getLayers()[j].neurons[i].weights}")
                    print(f"    bias: {self.getLayers()[j].neurons[i].bias}")
                neuronParams = [self.getLayers()[j].neurons[i].weights, self.getLayers()[j].neurons[i].bias]
                modelParams[j].append(neuronParams)
        return modelParams
                
    def getParamDifference(self, oldParams, newParams):
        modelDiff = []
        for i in range(len(oldParams)):
            modelDiff.append([])
            for j in range(len(oldParams[i])):
                layerWeightDiff = newParams[i][j][0] - oldParams[i][j][0]
                layerBiasDiff = newParams[i][j][1] - oldParams[i][j][1]
                modelDiff[i].append([layerWeightDiff, layerBiasDiff])

        for j in range(len(modelDiff)):
            for i in range(len(modelDiff[j])):
                print(f"layer: {j}, neuron: {i}")
                print(f"    weights change: {modelDiff[j][i][0]}")
                print(f"    bias change: {modelDiff[j][i][1]}")
        return modelDiff 