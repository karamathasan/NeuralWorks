import numpy as np
import pandas as pd

import model as m

from data_helper import dataSplit

def model1():
    data = pd.read_csv("datasets\course_engagement\online_course_engagement_data.csv")
    X = data[["TimeSpentOnCourse", "NumberOfVideosWatched", "NumberOfQuizzesTaken", "CompletionRate", "QuizScores"]].iloc[0].to_numpy()
    y = data[["CourseCompletion"]].iloc[0].to_numpy()

    rows = 6000
    predictor = data[["TimeSpentOnCourse", "NumberOfVideosWatched", "NumberOfQuizzesTaken", "CompletionRate", "QuizScores"]].iloc[0:rows]
    effector =  data[["CourseCompletion"]].iloc[0:rows]

    training_predictor, training_effector, testing_predictor, testing_effector = dataSplit(predictor, effector, 0.6, 0.8)

    model = m.Model(len(X),len(y), activationFunc= m.activation.Sigmoid(), lossFunc= m.loss.BinaryCrossEntropy(), optimizer=m.opt.RMSProp(0.1), metrics=m.metrics.Accuracy(), normalize_weights=True)
    model.addHiddenLayer(2)
    model.addHiddenLayer(2)

    # outputLayer = model.getLayerByIndex(1)
    # outputLayer.resetConnections(outputLayer.connections, m.activation.Sigmoid())

    old = model.getParams()
    model.test(testing_predictor, testing_effector)
    model.train(training_predictor, training_effector, testing_predictor, testing_effector, batch_size=1000, epochs=10)
    new = model.getParams()

    # y_pred = model.predict(X)
    model.test(testing_predictor, testing_effector)
    # model.getParamDifference(new, old)

# model 2
def model2():
    data = pd.read_csv(r"datasets\flight_prices\flight_dataset.csv")
    # print(data.head())
    
    X = data[["Total_Stops","Dep_hours", "Arrival_hours","Duration_hours"]].iloc[0].to_numpy()
    y = data[["Price"]].iloc[0].to_numpy()

    rows = 10000
    predictor = data[["Total_Stops","Dep_hours","Arrival_hours", "Duration_hours"]].iloc[0:rows]
    effector = data[["Price"]].iloc[0:rows]

    training_predictor, training_effector, testing_predictor, testing_effector = dataSplit(predictor, effector, 0.8, 0.8)
    assert (testing_predictor is not training_predictor)
    assert (testing_effector is not training_effector)

    model = m.Model(len(X),len(y), m.activation.Relu(), m.loss.SquaredError() , m.opt.RMSProp(0.03), metrics=m.metrics.AbsolutePercentError())
    model.addHiddenLayer(2)
    model.addHiddenLayer(2)
    model.addHiddenLayer(2)

    old = model.getParams(True)
    model.test(training_predictor,training_effector)
    model.train(training_predictor, training_effector, testing_predictor, testing_effector, 1500 ,15)

    model.test(testing_predictor, testing_effector)
    new = model.getParams(False)
    model.getParamDifference(new, old)

model1()