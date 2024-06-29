import numpy as np
import math
# import matplotlib as plt
import neuron as nrn
'''
a neural network has inputs, layers and outputs as objects
inputs must be a vector
layers are objects that must also include a number of neurons, activation functions
outputs must be a vector
learning rate, loss function, 
'''

# inputs
# X =
# 
input = np.array([1,2,5,1])
n = nrn.Neuron(1,1,"lin")
print(n.evaluate(input))


