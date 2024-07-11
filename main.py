import numpy as np
import pandas as pd

import model as m
import loss as l
from data_helper import dataSplit

def model1():
    data = pd.read_csv("datasets\course_engagement\online_course_engagement_data.csv")
    X = data[["TimeSpentOnCourse", "NumberOfVideosWatched", "NumberOfQuizzesTaken", "CompletionRate"]].iloc[0].to_numpy()
    y = data[["CourseCompletion"]].iloc[0].to_numpy()

    rows = 600
    predictor = data[["TimeSpentOnCourse", "NumberOfVideosWatched", "NumberOfQuizzesTaken", "CompletionRate"]].iloc[0:rows]
    effector =  data[["CourseCompletion"]].iloc[0:rows]

    training_predictor, training_effector, testing_predictor, testing_effector = dataSplit(predictor, effector, 0.6, 0.8)

    model = m.Model(len(X),len(y), activationFunc="sigmoid", lossFunc=l.BinaryCrossEntropy() , learningRate = 0.01)
    model.addHiddenLayer(4)
    model.addHiddenLayer(4)

    # outputLayer = model.getLayerByIndex(1)
    # outputLayer.resetConnections(outputLayer.connections, "sigmoid")
    # model.addHiddenLayer(4)

    old = model.getParams()
    model.test(testing_predictor, testing_effector)
    model.train(training_predictor, training_effector,'sgd',10)
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

    rows = 600
    predictor = data[["Total_Stops","Dep_hours","Arrival_hours", "Duration_hours"]].iloc[0:rows]
    effector = data[["Price"]].iloc[0:rows]

    training_predictor, training_effector, testing_predictor, testing_effector = dataSplit(predictor, effector, 0.6, 0.8)

    model = m.Model(len(X),len(y), "relu", 0.0001)
    model.addHiddenLayer(4)
    # model.addHiddenLayer(4)

    old = model.getParams(False)
    # model.test(testing_predictor, testing_effector)
    model.test(training_predictor,training_effector)
    model.train(training_predictor, training_effector,'sgd')
    new = model.getParams(False)

    # # y_pred = model.predict(X)
    model.test(training_predictor,training_effector)
    model.test(testing_predictor, testing_effector)
    # model.getParamDifference(new, old)

model1()