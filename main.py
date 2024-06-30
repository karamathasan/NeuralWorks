import numpy as np
import math


# import matplotlib as plt
import neuron as nrn
import layer 
import model as m
'''
a neural network has inputs, layers and outputs as objects
inputs must be a vector
layers are objects that must also include a number of neurons, activation functions
outputs must be a vector
learning rate, loss function, 
'''

# inputs


input = np.random.rand(16)
model = m.Model(16,2)
model.addHiddenLayer(4)
model.addHiddenLayer(4)
model.modelShape()
print(model.predict(input))
# print(layer.neurons[0].evaluate(input))


