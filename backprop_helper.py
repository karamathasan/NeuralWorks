import numpy as np

residuals = []
#potential source of error
seen = {}

def reset():
    global residuals
    seen.clear()
    residuals = []

def normClip(gradient, threshold = 1000):
    clippedGrad = 0
    if np.linalg.norm(gradient) > threshold:
        clippedGrad = threshold * (gradient/np.linalg.norm(gradient))
    else:
        return gradient
    return clippedGrad

def backpropagateWeight(layerIndex, neuronIndex, model):
    def dLdG(layerIndex, neuronIndex, model):
        derivativeName = getDerivativeName("L",layerIndex,neuronIndex,"g",layerIndex,neuronIndex)
        if (seen.get(derivativeName) is not None):
            return seen.get(derivativeName)
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
        derivativeName = getDerivativeName("L",layerIndex,neuronIndex,"v",layerIndex,neuronIndex)
        if (seen.get(derivativeName) is not None):
            return seen.get(derivativeName)
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

    # print(f"gradient: {gradient}")
    # print(f"gradient norm: {np.linalg.norm(gradient)}")

    if (model.gradient_clipping_magnitude is not None):
        gradient = normClip(gradient, model.gradient_clipping_magnitude)
    # print(f"layer: {layerIndex}, neuron: {neuronIndex}:: {gradient}")

    if (model.normalize_weights):
        g, v = getWeightParameters(neuron.weights)
        gGrad = dLdG(layerIndex, neuronIndex, model)
        vGrad = dLdV(layerIndex, neuronIndex, model)
        g = model.optimizer.evaluate(g,f"g: {layerIndex}, {neuronIndex}",gGrad)
        v = model.optimizer.evaluate(v,f"v: {layerIndex}, {neuronIndex}",vGrad)
        neuron.weights = g * (v/np.linalg.norm(v))
    else:

        # print(f"weight change{model.optimizer.evaluate(neuron.weights, gradient) - neuron.weights}")
        # print(f"neuron weights shape: {neuron.weights.shape}")
        # print(f"neuron weight gradient shape {gradient.shape}")
        # neuron.weights = neuron.weights - (gradient * model.getLearningRate())
        neuron.weights = model.optimizer.evaluate(neuron.weights, f"weight: {layerIndex}, {neuronIndex}", gradient)

def backpropagateBias(layerIndex, neuronIndex, model):
    gradient = normClip(dLdB(layerIndex, neuronIndex, model))
    layer = model.getLayerByIndex(layerIndex)
    neuron = layer.neurons[neuronIndex]
    # print(f"bias change{model.optimizer.evaluate(neuron.weights, gradient) - neuron.weights}")
    # neuron.bias = neuron.bias - (gradient * model.getLearningRate()) 
    neuron.bias = model.optimizer.evaluate(neuron.bias, f"bias: {layerIndex}, {neuronIndex}", gradient)

def getDerivativeName(numerator, numerL, numerN, denominator, denomL, denomN):
    return f"d{numerator}[{numerL}, {numerN}]/d{denominator}[{denomL},{denomN}]"

def backpropagate(model, Residuals):
    reset()
    global residuals
    residuals = Residuals
    allLayers = model.getLayers()

    for i in reversed(range(len(allLayers))):
        for j in range(len(allLayers[i].neurons)):
            # print(f"layer: {i} neuron: {j}")
            backpropagateBias(i,j,model) 
            backpropagateWeight(i,j,model)
    reset()


def dLdB(layerIndex, neuronIndex, model):
    derivativeName = (getDerivativeName("L",layerIndex,neuronIndex,"b",layerIndex,neuronIndex))
    # if (seen.get(derivativeName) is not None):
    #     return seen.get(derivativeName)
    
    value = dZdB(layerIndex, neuronIndex, model)  * dXdZ(layerIndex, neuronIndex, model) * dLdX(layerIndex, neuronIndex, model) 
    seen[derivativeName] = value
    # print(f"{derivativeName}: {seen[derivativeName]}")
    return value

def dLdW(layerIndex, neuronIndex, model):
    derivativeName = (getDerivativeName("L",layerIndex,neuronIndex,"w",layerIndex,neuronIndex))
    # if (seen.get(derivativeName) is not None):
    #     return seen.get(derivativeName)
    
    value =  dZdW(layerIndex, neuronIndex, model) * dXdZ(layerIndex, neuronIndex, model) * dLdX(layerIndex, neuronIndex, model) 
    seen[derivativeName] = value
    # print(f"{derivativeName}: {seen[derivativeName]}")
    return value

def dLdX(layerIndex, neuronIndex, model):
    derivativeName = (getDerivativeName("L",layerIndex,neuronIndex,"X",layerIndex,neuronIndex))
    # if (seen.get(derivativeName) is not None):
    #     return seen.get(derivativeName)
    value = 0
    if (layerIndex == len(model.hiddenLayers)):
        value = -2 * residuals[neuronIndex]
        # print()
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

def dLidX(residualIndex, layerIndex, neuronIndex, model):
    derivativeName = (getDerivativeName(f"L{residualIndex}",layerIndex,neuronIndex,"X",layerIndex,neuronIndex))
    # if (seen.get(derivativeName) is not None):
    #     return seen.get(derivativeName)
    value = 0
    if layerIndex == len(model.hiddenLayers):
        value = -2 * (residuals[residualIndex])
        seen[derivativeName] = value
        # print(f"{derivativeName}: {seen[derivativeName]}")
        return value

    sum = 0
    for i in range(len(model.getLayerByIndex(layerIndex+1).neurons)):
        sum += dLidX(residualIndex, layerIndex + 1, i, model) * dXdZ(layerIndex + 1, i, model) * dZidX(i,layerIndex + 1, neuronIndex, model)
    value = sum
    seen[derivativeName] = value
    # print(f"{derivativeName}: {seen[derivativeName]}")
    return value

def dXdZ(layerIndex, neuronIndex, model):
    derivativeName = getDerivativeName("X",layerIndex,neuronIndex,"Z",layerIndex,neuronIndex)
    # if (seen.get(derivativeName) is not None):
    #     return seen.get(derivativeName)
    layer = model.getLayerByIndex(layerIndex)
    neuron = layer.neurons[neuronIndex]

    value = neuron.activationFunctionDerivative()
    seen[derivativeName] = value
    # print(f"{derivativeName}: {seen[derivativeName]}")
    return value

def dZidX(zIndex, zLayerIndex, xNeuronIndex, model):
    # del Z[L,n]/ del X[L-1,n] 
    derivativeName = getDerivativeName("Z",zLayerIndex,zIndex,"X", zLayerIndex-1, xNeuronIndex)
    # if (seen.get(derivativeName) is not None):
    #     return seen.get(derivativeName)
    layer = model.getLayerByIndex(zLayerIndex)
    neuron = layer.neurons[zIndex]

    value = 0
    value = neuron.weights[xNeuronIndex]
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

def dZdW(layerIndex, neuronIndex, model):
    # del Z[L,n]/ del w[L,n]
    derivativeName = getDerivativeName("Z",layerIndex,neuronIndex,"w",layerIndex,neuronIndex)
    # if (seen.get(derivativeName) is not None):
    #     return seen.get(derivativeName)
    layer = model.getLayerByIndex(layerIndex)
    neuron = layer.neurons[neuronIndex]

    value = 0
    value = neuron.input
    # for input in neuron.input:
    #     value += input
    # print(f"dZdW: {value} at {layerIndex, neuronIndex, model}" )
    seen[derivativeName] = value
    # print(f"{derivativeName}: {seen[derivativeName]}")
    return value

