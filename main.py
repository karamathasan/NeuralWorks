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

    model = m.Model(len(X),len(y), activationFunc= m.activation.Sigmoid(), lossFunc= m.loss.BinaryCrossEntropy(), optimizer=m.opt.Adam(0.003))
    model.addHiddenLayer(2)
    model.addHiddenLayer(2)

    # outputLayer = model.getLayerByIndex(1)
    # outputLayer.resetConnections(outputLayer.connections, m.activation.Sigmoid())

    old = model.getParams()
    model.test(testing_predictor, testing_effector)
    model.train(training_predictor, training_effector, 60 ,25)
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

    model = m.Model(len(X),len(y), m.activation.Relu(), m.loss.SquaredError() , m.opt.Adam(0.3))
    model.addHiddenLayer(4)
    # model.addHiddenLayer(4)

    old = model.getParams(False)
    # model.test(testing_predictor, testing_effector)
    model.test(training_predictor,training_effector)
    model.train(training_predictor, training_effector, "full-batch" ,25)
    new = model.getParams(False)

    # # y_pred = model.predict(X)
    # model.test(training_predictor,training_effector)
    model.test(testing_predictor, testing_effector)
    # model.getParamDifference(new, old)

model1()