import numpy as np
import pandas as pd

import loss 
import activation
import optimizer as opt 

import layer 
from data_helper import shuffle
from backprop_helper import backpropagate

class Model():
    '''
    inputSize refers to the length of the vector associated with the input layer. likewise for the outputSize
    the output layer SHOULD NOT use a layer object, and instead should use a vector
    '''
    def __init__(self, inputSize = 1, outputSize = 1, activationFunc = activation.Relu(), lossFunc = loss.SquaredError(), optimizer = opt.SGD(0.001)):
        self.inputSize = inputSize
        self.hiddenLayers = []
        # self.learningRate = learningRate
        
        assert isinstance(activationFunc, activation.ActivationFunction), "ACTIVATION FUNCTION PARAMETER IS NOT OF TYPE ACTIVATIONFUNCTION"
        assert isinstance(lossFunc, loss.LossFunction), "LOSS FUNCTION PARAMETER IS NOT OF TYPE LOSSFUNCTION"
        assert isinstance(optimizer, opt.Optimizer), "OPTIMIZER PARAMETER IS NOT OF TYPE OPTIMIZER"
 
        self.defaultActivation = activationFunc # a default activation is made so that newly added activation functions will be set to this activationFunction
        self.lossFunc = lossFunc
        self.optimizer = optimizer
        self.outputLayer = layer.Layer(outputSize, inputSize, activationFunc)

    def predict(self, input):
        allLayers = self.getLayers()
        output = input
        for i in range(len(allLayers)):
            currentLayer = allLayers[i]
            output = currentLayer.evaluate(output)
        return output

    def train(self, predictor_data: pd.DataFrame, effector_data: pd.DataFrame, batch_size, epochs = 1):
        '''
        train the model 

        Args:
            predictor_data: dataframe with the input data to predict an output
            effector_data: dataframe with the corresponding true outputs to the input
            batch_size: int or string referring to the number of datapoints used to accumulate error for a backward pass
                >>>accepted strings: "full-batch", "stochastic"
        '''
        assert (predictor_data.shape[0] == effector_data.shape[0]) 
        

        for j in range(epochs):    
            print(f"epoch {j + 1} begin")
            if (type(batch_size) == int):
                assert (batch_size > 1), (f"INVALID INPUT FOR BATCH_SIZE, {batch_size}")
                residuals = 0
                for i in range(len(predictor_data)):
                    row = predictor_data.iloc[i].to_numpy()   
                    true = effector_data.iloc[i].to_numpy()
                    prediction = self.predict(row)
                    residuals += self.calculateResidualsArray(true,prediction)
                    if(i % batch_size == 0 and i != 0):
                        backpropagate(self,residuals)
                        residuals = 0

            elif batch_size == "full-batch":
                residuals = 0
                for i in range(len(predictor_data)):
                    row = predictor_data.iloc[i].to_numpy()
                    prediction = self.predict(row)
                    true = effector_data.iloc[i].to_numpy()
                    residuals += self.calculateResidualsArray(true, prediction)
                backpropagate(self,residuals)

            elif batch_size == "stochastic":
                for i in range(len(predictor_data)):
                    row = predictor_data.iloc[i].to_numpy()
                    prediction = self.predict(row)
                    true = effector_data.iloc[i].to_numpy()
                    print(f"iteration: {j,i}")
                    print(f"    prediction: {prediction}")
                    print(f"    true: {true}")
                    # print(f"    squared diff: {loss.SquaredError().evaluateAsArray(true, prediction)}")
                    residuals = self.calculateResidualsArray(true, prediction)
                    print(f"    residual: {residuals}")
                    backpropagate(self,residuals)
            else:
                return ValueError(f"INVALID INPUT FOR BATCH_SIZE, {batch_size}")
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
    
    def getLearningRate(self):
        return self.optimizer.learningRate