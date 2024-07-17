import matplotlib.pyplot as plt

class Log():
    def __init__(self, model):
        # self.weights = []
        # self.biases = []
        self.losses = []
        self.preformance = []
        self.predictions = []
        self.truths = []
        
        self.metric = model.metrics
        self.lossFunc = model.lossFunc

    def update(self, y_true, y_pred):
        self.predictions.append(y_pred)
        self.truths.append(y_true)

        self.losses.append(self.lossFunc.evaluateAsSum(y_true,y_pred))
        self.preformance.append(self.metric.evaluate(y_true, y_pred))
    
    def addLossPlot(self):
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.plot(self.losses, label = "Loss vs. Iterations")

    def addPreformancePlot(self):
        plt.xlabel("iteration")
        plt.ylabel("preformance")
        plt.plot(self.preformance, label = "Preformance vs. Iterations")

    def graph(self):
        plt.legend()
        plt.show()

class EpochLog(Log):
    def __init__(self,model):
        super().__init__(model)
        self.validation_losses = []
        self.validation_preformance = []
        self.validation_predictions = []
        self.validation_truths = []
    
    def updateValidation(self, validation_true, validation_prediction):
        self.validation_predictions.append(validation_prediction)
        self.validation_truths.append(validation_true)

        self.validation_losses.append(self.lossFunc.evaluateAsSum(validation_true,validation_prediction))
        self.validation_preformance.append(self.metric.evaluate(validation_true, validation_prediction))

    def addLossPlot(self):
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(self.losses, label = "Loss vs. Epoch")

    def addValidationLossPlot(self):
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(self.validation_losses, label = "Validation Loss vs. Epoch")

    def addPreformancePlot(self):
        plt.xlabel("epoch")
        plt.ylabel(self.metric.getName())
        plt.plot(self.preformance, label = "Preformance vs. Epoch")
    
    def addValidationPreformancePlot(self):
        plt.xlabel("epoch")
        plt.ylabel(self.metric.getName())
        plt.plot(self.validation_preformance, label = "Validation Preformance vs. Epoch")