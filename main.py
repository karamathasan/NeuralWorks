import numpy as np
import pandas as pd

import model as m
from data_helper import dataSplit

data = pd.read_csv("datasets\course_engagement\online_course_engagement_data.csv")
X = data[["TimeSpentOnCourse", "NumberOfVideosWatched", "NumberOfQuizzesTaken", "CompletionRate"]].iloc[0].to_numpy()
y = data[["CourseCompletion"]].iloc[0].to_numpy()

rows = 600
predictor = data[["TimeSpentOnCourse", "NumberOfVideosWatched", "NumberOfQuizzesTaken", "CompletionRate"]].iloc[0:rows]
effector =  data[["CourseCompletion"]].iloc[0:rows]

training_predictor, training_effector, testing_predictor, testing_effector = dataSplit(predictor, effector, 0.6, 0.8)

model = m.Model(len(X),len(y), "sigmoid", 0.01)
model.addHiddenLayer(4)
# model.addHiddenLayer(4)

# old = model.getParams()
model.test(testing_predictor, testing_effector)
model.train(training_predictor, training_effector,'mini-batch')
# new = model.getParams()

# y_pred = model.predict(X)
model.test(testing_predictor, testing_effector)
# model.getParamDifference(new, old)
