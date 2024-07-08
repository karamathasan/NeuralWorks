import numpy as np

import layer as l
import model as m
import neuron as nrn

residuals = []
residualSum = 0
ssr = 0
#potential source of error
seen = {}

# def backpropagateWeight(layerIndex, neuronIndex, model):
#     gradient = -dLdW(layerIndex, neuronIndex, model)
#     layer = model.getLayerByIndex(layerIndex)
#     neuron = layer.neurons[neuronIndex]
#     # print(gradient)
#     neuron.weights += -gradient * model.learningRate

def backpropagateWeight(layerIndex, neuronIndex, model, normalizeWeights = False):
    def dLdG(layerIndex, neuronIndex, model):
        derivativeName = getDerivativeName("Z",layerIndex,neuronIndex,"w",layerIndex,neuronIndex)
        # print(derivativeName)
        # if (seen.get(derivativeName) is not None):
        #     return seen.get(derivativeName)
        layer = model.getLayerByIndex(layerIndex)
        neuron = layer.neurons[neuronIndex]

        g,v = getWeightParameters(neuron.weights)
        value = 0
        value = dLdW(layerIndex,neuronIndex, model) * v
        # print(f"dZdW: {value} at {layerIndex, neuronIndex, model}" )
        seen[derivativeName] = value
        # print(f"{derivativeName}: {seen[derivativeName]}")
        return value
    def dLdV(layerIndex, neuronIndex, model):
        derivativeName = getDerivativeName("Z",layerIndex,neuronIndex,"w",layerIndex,neuronIndex)
        # print(derivativeName)
        # if (seen.get(derivativeName) is not None):
        #     return seen.get(derivativeName)
        layer = model.getLayerByIndex(layerIndex)
        neuron = layer.neurons[neuronIndex]

        g,v = getWeightParameters(neuron.weights)
        value = 0
        value = g / np.linalg.norm(v) - (dLdG(layerIndex,neuronIndex, model)/ (np.linalg.norm(v)*np.linalg.norm(v)))  * v 
        # print(f"dZdW: {value} at {layerIndex, neuronIndex, model}" )
        seen[derivativeName] = value
        # print(f"{derivativeName}: {seen[derivativeName]}")
        return value
    def getWeightParameters(weight):
        # w = g * v/||v||
        # ||w|| = g
        # w / g = v/||v||
        g = np.linalg.norm(weight)
        v = weight / g
        return g,v
    
    layer = model.getLayerByIndex(layerIndex)
    neuron = layer.neurons[neuronIndex]

    gradient = dLdW(layerIndex, neuronIndex, model)
    # print(f"layer: {layerIndex}, neuron: {neuronIndex}:: {gradient}")

    if (normalizeWeights):
        g, v = getWeightParameters(neuron.weights)
        gGrad = dLdG(layerIndex, neuronIndex, model)
        vGrad = dLdV(layerIndex, neuronIndex, model)
        g = g -gGrad * model.learningRate
        v = v -vGrad * model.learningRate
        neuron.weights = g * (v/np.linalg.norm(v))
    else:
        # original = neuron.weights
        neuron.weights = neuron.weights - gradient * model.learningRate
        # print(f"update value: {gradient * model.learningRate}")
        # print(f"weight difference: {original - neuron.weights}")

def backpropagateBias(layerIndex, neuronIndex, model):
    gradient = dLdB(layerIndex, neuronIndex, model)
    layer = model.getLayerByIndex(layerIndex)
    neuron = layer.neurons[neuronIndex]
    # print(gradient)
    neuron.bias -= gradient * model.learningRate

def getDerivativeName(numerator, numerL, numerN, denominator, denomL, denomN):
    return f"d{numerator}[{numerL}, {numerN}]/d{denominator}[{denomL},{denomN}]"

def backpropagate(model, Residuals):
    global residuals
    global ssr
    global residualSum
    residuals = Residuals
    allLayers = model.getLayers()
    for residual in residuals:
        residualSum += residual
        ssr += residual * residual
    for i in reversed(range(len(allLayers))):
        for j in range(len(allLayers[i].neurons)):
            # print(f"layer: {i} neuron: {j}")
            backpropagateBias(i, j,model) 
            backpropagateWeight(i,j,model, True)
            # for w in range(len(allLayers[i].neurons[j].weights)):
            #     backpropagateWeight(i, j, w, model)

def dLdB(layerIndex, neuronIndex, model):
    derivativeName = (getDerivativeName("L",layerIndex,neuronIndex,"b",layerIndex,neuronIndex))
    # if (seen.get(derivativeName) is not None):
    #     return seen.get(derivativeName)
    
    value = dZdB(layerIndex, neuronIndex, model)  * dXdZ(layerIndex, neuronIndex, model) * dLdX(layerIndex, neuronIndex, model) 
    seen[derivativeName] = value
    # print(f"{derivativeName}: {seen[derivativeName]}")
    return value

# def dLdW(layerIndex, neuronIndex, model):
#     derivativeName = (getDerivativeName("L",layerIndex,neuronIndex,"w",layerIndex,neuronIndex))
#     # print(derivativeName)
#     # if (seen.get(derivativeName) is not None):
#     #     return seen.get(derivativeName)
    
#     value =  dZdW(layerIndex, neuronIndex, model) * dXdZ(layerIndex, neuronIndex, model) * dLdX(layerIndex, neuronIndex, model) 
#     seen[derivativeName] = value
#     # print(f"{derivativeName}: {seen[derivativeName]}")
#     return value

def dLdW(layerIndex, neuronIndex, model):
    derivativeName = (getDerivativeName("L",layerIndex,neuronIndex,"w",layerIndex,neuronIndex))
    # print(derivativeName)
    # if (seen.get(derivativeName) is not None):
    #     return seen.get(derivativeName)
    
    value =  dZdW(layerIndex, neuronIndex, model) * dXdZ(layerIndex, neuronIndex, model) * dLdX(layerIndex, neuronIndex, model) 
    seen[derivativeName] = value
    # print(f"{derivativeName}: {seen[derivativeName]}")
    return value

def dLdX(layerIndex, neuronIndex, model):
    derivativeName = (getDerivativeName("L",layerIndex,neuronIndex,"X",layerIndex,neuronIndex))
    # print(derivativeName)
    # if (seen.get(derivativeName) is not None):
    #     return seen.get(derivativeName)
    value = 0
    if (layerIndex == len(model.hiddenLayers)):
        # L is the sum of the squared residuals
        # L = (y-y*)^2
        # y* = sigma(wx+b) = activation of the output layer
        value = -2*(residualSum)
        seen[derivativeName] = value
        # print(f"{derivativeName}: {seen[derivativeName]}")
        return value
    else:
        sum = 0
        for i in range(len(residuals)):
        # for i in range(len(model.getLayerByIndex(layerIndex+1).neurons)):
            sum += dLidX(i,layerIndex,neuronIndex, model)
        value = sum
        seen[derivativeName] = value
        # print(f"{derivativeName}: {seen[derivativeName]}")
        return value

def dLidX(outputIndex, layerIndex, neuronIndex, model):
    derivativeName = (getDerivativeName(f"L{outputIndex}",layerIndex,neuronIndex,"X",layerIndex,neuronIndex))
    # print(derivativeName)
    # if (seen.get(derivativeName) is not None):
    #     return seen.get(derivativeName)
    value = 0
    if layerIndex == len(model.hiddenLayers):
        value = -2 * (residuals[outputIndex])
        seen[derivativeName] = value
        # print(f"{derivativeName}: {seen[derivativeName]}")
        return value
    # value = dLidZ(outputIndex, layerIndex + 1, neuronIndex, model) * dZdX(layerIndex + 1, neuronIndex, model)
    sum = 0
    for i in range(len(model.getLayerByIndex(layerIndex+1).neurons)):
        sum += dLidZ(outputIndex, layerIndex + 1, i, model) * dZdX(layerIndex + 1, i, model)
    value = sum
    seen[derivativeName] = value
    # print(f"{derivativeName}: {seen[derivativeName]}")
    return value

def dLidZ(outputIndex, layerIndex, neuronIndex, model):
    derivativeName = (getDerivativeName(f"L{outputIndex}",layerIndex,neuronIndex,"Z",layerIndex,neuronIndex))
    # print(derivativeName)
    # if (seen.get(derivativeName) is not None):
    #     return seen.get(derivativeName)
    value = 0
    value = dLidX(outputIndex, layerIndex, neuronIndex, model) * dXdZ(layerIndex, neuronIndex, model)
    seen[derivativeName] = value
    # print(f"{derivativeName}: {seen[derivativeName]}")
    return value

def dXdZ(layerIndex, neuronIndex, model):
    derivativeName = getDerivativeName("X",layerIndex,neuronIndex,"Z",layerIndex,neuronIndex)
    # print(derivativeName)
    # if (seen.get(derivativeName) is not None):
    #     return seen.get(derivativeName)
    layer = model.getLayerByIndex(layerIndex)
    neuron = layer.neurons[neuronIndex]

    value = neuron.activationFunctionDerivative()
    seen[derivativeName] = value
    # print(f"{derivativeName}: {seen[derivativeName]}")
    return value

def dZdX(layerIndex, neuronIndex, model):
    # del Z[L,n]/ del X[L-1,n] 
    derivativeName = getDerivativeName("Z",layerIndex,neuronIndex,"X",layerIndex-1,neuronIndex)
    # print(derivativeName)
    # if (seen.get(derivativeName) is not None):
    #     return seen.get(derivativeName)
    layer = model.getLayerByIndex(layerIndex)
    neuron = layer.neurons[neuronIndex]

    # value = neuron.weights
    value = 0
    for weight in neuron.weights:
        value += weight
    seen[derivativeName] = value
    # print(f"{derivativeName}: {seen[derivativeName]}")
    # print(f"dZdX: {value} at {layerIndex, neuronIndex, model}" )
    return value

def dZdB(layerIndex, neuronIndex, model):
    # del Z[L,n]/ del b[L,n] 
    derivativeName = getDerivativeName("Z",layerIndex,neuronIndex,"b",layerIndex,neuronIndex)
    # if (seen.get(derivativeName) is not None):
    #     return seen.get(derivativeName)
    value = 1
    seen[derivativeName] = value
    # print(f"{derivativeName}: {seen[derivativeName]}")
    return value

# def dZdW(layerIndex, neuronIndex, model):
#     # del Z[L,n]/ del w[L,n]
#     derivativeName = getDerivativeName("Z",layerIndex,neuronIndex,"w",layerIndex,neuronIndex)
#     # print(derivativeName)
#     # if (seen.get(derivativeName) is not None):
#     #     return seen.get(derivativeName)
#     layer = model.getLayerByIndex(layerIndex)
#     neuron = layer.neurons[neuronIndex]

#     value = 0
#     # value = neuron.input
#     for input in neuron.input:
#         value += input
#     # print(f"dZdW: {value} at {layerIndex, neuronIndex, model}" )
#     seen[derivativeName] = value
#     # print(f"{derivativeName}: {seen[derivativeName]}")
#     return value

def dZdW(layerIndex, neuronIndex, model):
    # del Z[L,n]/ del w[L,n]
    derivativeName = getDerivativeName("Z",layerIndex,neuronIndex,"w",layerIndex,neuronIndex)
    # print(derivativeName)
    # if (seen.get(derivativeName) is not None):
    #     return seen.get(derivativeName)
    layer = model.getLayerByIndex(layerIndex)
    neuron = layer.neurons[neuronIndex]

    value = 0
    # value = neuron.input[weightIndex]
    value = neuron.input
    # print(f"dZdW: {value} at {layerIndex, neuronIndex, model}" )
    seen[derivativeName] = value
    # print(f"{derivativeName}: {seen[derivativeName]}")
    return value

