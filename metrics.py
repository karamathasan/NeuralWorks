import numpy as np

class Metric():
    def __init__(self, func):
        self.metricEvaluation = func

    def evaluate(self, y_true, y_pred):
        return self.metricEvaluation(y_true, y_pred)
    
    # def evaluateMean(self, y_true, y_pred):
    #     assert len(y_true) == len(y_pred)
    #     num_correct = 0
    #     for i in range(len(y_true)):
    #         num_correct += self.metricEvaluation(y_true[i], y_pred[i])
    #     return num_correct / len(y_true)
    
    def getName(self):
        return NotImplementedError
    

class SquaredError(Metric):
    def __init__(self):
        func = lambda y_true, y_pred : (y_true-y_pred)**2
        super().__init__(func)

    def getName(self):
        return "Squared Error"

class Accuracy(Metric):
    def __init__(self):
        # func = lambda y_true, y_pred : 1 if (np.argmax(y_pred) == np.argmax(y_true)) else 0
        func = lambda y_true, y_pred : 1 if (y_true.all() == (self.encodeOneHot(y_pred)).all()) else 0

        super().__init__(func)

    def encodeOneHot(self, vec):
        max_index = np.argmax(vec)
        one_hot = np.zeros(len(vec))
        one_hot[max_index] = 1
        return one_hot
    
    def getName(self):
        return "Accuracy"

class AbsolutePercentError(Metric):
    def __init__(self):
        func = lambda y_true, y_pred : 100 * np.abs(y_true - y_pred) / y_true
        super().__init__(func)

    def getName(self):
        return "Absolute Percent Error"
