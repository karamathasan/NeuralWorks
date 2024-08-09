class Trainer():
    def __init__(self, model):
        self.model = model
        pass

    def backpropagate(self, residuals):
        # reset()
        self.residuals = residuals
        allLayers = self.model.getLayers()

        gradients = {}
        for i in reversed(range(len(allLayers))):
            for j in range(len(allLayers[i].neurons)):
                # print(f"layer: {i} neuron: {j}")
                '''
                for each tunable paramater, take the necessary chain rules based on the loss, activations and parameter itself and then save them into a dictionary
                '''
                for parameter in allLayers[i].neurons[j].tunableParams:
                    gradients[f"layer: {i} neuron: {j}"] = parameter.dZdTheta()
                pass


    # def backwardpass(self)