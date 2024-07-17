import numpy as np
class Optimizer():
    def __init__(self, func, learningRate):
        self.func = func
        self.learningRate = learningRate

    def evaluate(self, input, position, gradient):
        '''
        evaluate the new parameter at a specific neuron
        
        Args: 
            input: the input given to the optimizer/the parameter's value currently
            position: the position of the paramater as a string "paramaterType: layer, neuron"
                -> "parameterType" denotes weight or bias, or g or v when weight normalizing
                -> ex. f"weight: 0, 1" denotes the weight at layer 0, neuron 1, both indexed from 0
            gradient: the gradient of the paramater being updated
        '''
        return self.func(input, position, gradient)
    
    def reset(self):
        raise NotImplementedError
    
class SGD(Optimizer):
    def __init__(self, learningRate):
        func = lambda X, position, nablaX : X - learningRate * nablaX
        super().__init__(func, learningRate)
    
class SGD_Momentum(Optimizer):
    def __init__(self, learningRate, beta = 0.9):
        self.beta = beta
        func = lambda X, position, nablaX : X - learningRate * self.updateVelocity(position, nablaX)
        super().__init__(func, learningRate)
        self.velocities = {}

    def updateVelocity(self, position, gradient):
        newVelocity = self.beta * self.velocities.setdefault(position, 0) + (1-self.beta) * gradient
        self.velocities[position] = newVelocity
        return newVelocity
    
    def evaluate(self, input, position, gradient):

        output = self.func(input, position, gradient)
        # output = super().evaluate(input, position, gradient)
        return output
    
# class Nesterov(Optimizer):
#     def __init__(self, learningRate, beta):
#         self.beta = beta
#         func = lambda X, nablaX, rate : X - rate * self.updateVelocity(nablaX)
#         super().__init__(func, learningRate)
#         self.previousVelocity = 0

#     def updateVelocity(self, gradient):
#         newVelocity = self.beta * self.previousVelocity + (1-self.beta) * gradient
#         self.previousVelocity = newVelocity
#         return newVelocity
    
#     def evaluate(self, input, gradient):
#         output = super().evaluate(input, gradient)
#         return output   
    
class RMSProp(Optimizer):
    def __init__(self, learningRate, beta = 0.9):
        self.beta = beta
        func = lambda X, position, nablaX: X - learningRate * (nablaX/np.sqrt(self.updateVelocity(position, nablaX) + 1e-8))
        super().__init__(func, learningRate)
        self.velocities = {}

    def updateVelocity(self, position, gradient):
        newVelocity = self.beta * self.velocities.setdefault(position, 0) + (1-self.beta) * gradient * gradient
        self.velocities[position] = newVelocity
        return newVelocity
    
    def evaluate(self, input, position, gradient):
        output = super().evaluate(input, position, gradient)
        return output
    
class AdaGrad(Optimizer):
    def __init__(self, learningRate, beta = 0.9):
        self.beta = beta
        func = lambda X, position, nablaX : X - learningRate * (nablaX/np.sqrt(self.updateVelocity(position, nablaX) + 1e-8))
        super().__init__(func, learningRate)
        self.velocities = {}

    def updateVelocity(self, position, gradient):
        newVelocity = self.beta * self.velocities.setdefault(position, 0) + gradient * gradient
        self.velocities[position] = newVelocity
        return newVelocity
    
    def evaluate(self, input, position, gradient):
        output = super().evaluate(input, position, gradient)
        return output

class Adam(Optimizer):
    def __init__(self, learningRate, beta_1 = 0.9, beta_2 = 0.99):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        func = lambda X, position, nablaX : X - learningRate * (self.correctMoment(position, self.updateMoment(position,nablaX))/np.sqrt(self.correctVelocity(position, self.updateVelocity(position, nablaX)) + 1e-8))
        super().__init__(func, learningRate)
        self.velocities = {}
        self.moments = {}
        self.iterations = {}

    def updateMoment(self, position, gradient):
        newMoment = self.beta_1 * self.moments.setdefault(position, 1) + (1-self.beta_1) * gradient
        self.previousMoment = newMoment
        return newMoment

    def updateVelocity(self, position, gradient):
        newVelocity = self.beta_2 * self.velocities.setdefault(position, 1) + (1-self.beta_2) * gradient * gradient
        self.previousVelocity = newVelocity
        return newVelocity
    
    def correctMoment(self, position, moment):
        return moment / (1 - self.beta_1 ** self.iterations.setdefault(position, 1))
    
    def correctVelocity(self, position, velocity):
        return velocity / (1 - self.beta_2 ** self.iterations.setdefault(position, 1))
    
    def evaluate(self, input, position, gradient):
        self.iterations.setdefault(position, 1)
        output = super().evaluate(input, position, gradient)
        self.iterations[position] += 1 
        return output
    
    def reset(self):
        self.moments = {}
        self.velocities = {}
        self.iterations = {}