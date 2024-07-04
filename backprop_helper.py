import numpy as np

import layer as l
import model as m
import neuron as nrn

residuals = []
residualSum = 0
ssr = 0
seen = {}

def backpropagateWeight(layerIndex, neuronIndex, model):
    gradient = -dLdW(layerIndex, neuronIndex, model)
    layer = model.getLayerByIndex(layerIndex)
    neuron = layer.neurons[neuronIndex]
    # print(gradient)
    print(f"gradient shape: {gradient.shape}")
    print(f"weights shape: {neuron.weights.shape}")
    # print(neuron.weights)
    # print(neuron.weights)
    neuron.weights += gradient

def backpropagateBias(layerIndex, neuronIndex, model):
    gradient = -dLdW(layerIndex, neuronIndex, model)
    layer = model.getLayerByIndex(layerIndex)
    neuron = layer.neurons[neuronIndex]
    neuron.bias += -gradient * model.learningRate

def getDerivativeName(numerator, numerL, numerN, denominator, denomL, denomN):
    return f"d{numerator}[{numerL}, {numerN}]/d{denominator}[{denomL},{denomN}]"

def backpropagate(model, Residuals):
    global residuals
    global ssr
    global residualSum
    residuals = Residuals
    # layersReversed = []
    # layersReversed.append(model.outputLayer)
    # layersReversed.extend(model.getHiddenLayersReversed())

    allLayers = model.getLayers()

    for residual in residuals:
        residualSum += residual
        ssr += residual * residual
    # for iReversed in range(len(layersReversed)):
    #     for jReversed in range(len(layersReversed[iReversed].neurons)):
    #         i = len(layersReversed) - 1 - iReversed
    #         j = len(layersReversed[i].neurons) - 1 - jReversed
    #         print(j,i)
    #         backpropagateWeight(j,i,model)
    #         backpropagateBias(j,i,model)
    for i in range(len(allLayers)):
        iReversed = len(allLayers) - 1 - i
        for j in range(len(allLayers[iReversed].neurons)):
            # print(j,iReversed)
            backpropagateWeight(j,iReversed,model)
            backpropagateBias(j,iReversed,model) 


def dLdB(layerIndex, neuronIndex, model):
    derivativeName = (getDerivativeName("L",layerIndex,neuronIndex,"b",layerIndex,neuronIndex))
    print(derivativeName)
    if (seen.get(derivativeName) is not None):
        return seen.get(derivativeName)
    
    value = dLdX(layerIndex, neuronIndex, model) * dXdZ(layerIndex, neuronIndex, model) * dZdB(layerIndex, neuronIndex, model)
    seen[derivativeName] = value
    return value

def dLdW(layerIndex, neuronIndex, model):
    derivativeName = (getDerivativeName("L",layerIndex,neuronIndex,"w",layerIndex,neuronIndex))
    print(derivativeName)
    if (seen.get(derivativeName) is not None):
        return seen.get(derivativeName)
    
    value = dLdX(layerIndex, neuronIndex, model) * dXdZ(layerIndex, neuronIndex, model) * dZdW(layerIndex, neuronIndex, model)
    seen[derivativeName] = value
    return value

def dLdX(layerIndex, neuronIndex, model):
    derivativeName = (getDerivativeName("L",layerIndex,neuronIndex,"X",layerIndex,neuronIndex))
    print(derivativeName)
    if (seen.get(derivativeName) is not None):
        return seen.get(derivativeName)
    value = 0
    if (layerIndex == len(model.hiddenLayers)):
        # L is the sum of the squared residuals
        # L = (y-y*)^2
        # y* = sigma(wx+b) = activation of the output layer
        value = -2*(residualSum)
        seen[derivativeName] = value
        return value
    else:
        sum = 0
        # for residual in residuals:
        for i in range(len(residuals)):
            sum += dLidX(i,layerIndex,neuronIndex, model)
        value = sum
        seen[derivativeName] = value
        return value

def dLidX(outputIndex, layerIndex, neuronIndex, model):
    derivativeName = (getDerivativeName(f"L{outputIndex}",layerIndex,neuronIndex,"X",layerIndex,neuronIndex))
    print(derivativeName)
    if (seen.get(derivativeName) is not None):
        return seen.get(derivativeName)
    if layerIndex == len(model.hiddenLayers):
        return -2 * (residuals[outputIndex])
    value = dLidZ(outputIndex, layerIndex + 1, neuronIndex, model) * dZdX(layerIndex + 1, neuronIndex, model)
    seen[derivativeName] = value
    return value

def dLidZ(outputIndex, layerIndex, neuronIndex, model):
    derivativeName = (getDerivativeName(f"L{outputIndex}",layerIndex,neuronIndex,"Z",layerIndex,neuronIndex))
    print(derivativeName)
    if (seen.get(derivativeName) is not None):
        return seen.get(derivativeName)
    
    value = dLidX(outputIndex, layerIndex+1, neuronIndex, model) * dXdZ(outputIndex,layerIndex+1, neuronIndex, model)
    seen[derivativeName] = value
    return value

def dXdZ(layerIndex, neuronIndex, model):
    derivativeName = getDerivativeName("X",layerIndex,neuronIndex,"Z",layerIndex,neuronIndex)
    print(derivativeName)
    if (seen.get(derivativeName) is not None):
        return seen.get(derivativeName)
    layer = model.getLayerByIndex(layerIndex)
    neuron = layer.neurons[neuronIndex]

    value = neuron.activationFunctionDerivative()
    seen[derivativeName] = value
    return value

def dZdX(layerIndex, neuronIndex, model):
    # del Z[L,n]/ del X[L-1,n] 
    derivativeName = getDerivativeName("Z",layerIndex,neuronIndex,"X",layerIndex-1,neuronIndex)
    print(derivativeName)
    if (seen.get(derivativeName) is not None):
        return seen.get(derivativeName)
    layer = model.getLayerByIndex(layerIndex)
    neuron = layer.neurons[neuronIndex]

    value = neuron.weights
    seen[derivativeName] = value
    # print(f"dZdX: {value} at {layerIndex, neuronIndex, model}" )
    return value

def dZdB(layerIndex, neuronIndex, model):
    # del Z[L,n]/ del b[L,n] 
    derivativeName = getDerivativeName("Z",layerIndex,neuronIndex,"b",layerIndex,neuronIndex)
    if (seen.get(derivativeName) is not None):
        return seen.get(derivativeName)
    value = 1
    seen[derivativeName] = value
    return value

def dZdW(layerIndex, neuronIndex, model):
    # del Z[L,n]/ del w[L,n]
    derivativeName = getDerivativeName("Z",layerIndex,neuronIndex,"w",layerIndex,neuronIndex)
    print(derivativeName)
    if (seen.get(derivativeName) is not None):
        return seen.get(derivativeName)
    layer = model.getLayerByIndex(layerIndex)
    neuron = layer.neurons[neuronIndex]

    value = neuron.input
    # print(f"dZdW: {value} at {layerIndex, neuronIndex, model}" )
    seen[derivativeName] = value
    return value

