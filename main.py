import numpy as np
import pandas as pd

import model as m
from data_helper import dataSplit

'''

'''


'''
a neural network has inputs, layers and outputs as objects
inputs must be a vector
layers are objects that must also include a number of neurons, activation functions
outputs must be a vector
learning rate, loss function, 
'''


data = pd.read_csv("datasets\course_engagement\online_course_engagement_data.csv")
X = data[["TimeSpentOnCourse", "NumberOfVideosWatched", "NumberOfQuizzesTaken", "CompletionRate"]].iloc[0].to_numpy()
y = data[["CourseCompletion"]].iloc[0].to_numpy()

rows = 600
predictor = data[["TimeSpentOnCourse", "NumberOfVideosWatched", "NumberOfQuizzesTaken", "CompletionRate"]].iloc[0:rows]
effector =  data[["CourseCompletion"]].iloc[0:rows]

training_predictor, training_effector, testing_predictor, testing_effector = dataSplit(predictor, effector)


# X = np.random.rand(len(X))

model = m.Model(len(X),len(y), "sigmoid")
model.addHiddenLayer(2)
# model.addHiddenLayer(4)

# print(model.train(training_predictor.to_numpy(), training_effector.to_numpy()))
# print(model.train(training_predictor, training_effector))
# model.modelShape()
y_pred = model.predict(X)
model.train(training_predictor, training_effector)
# model.modelShape()
# print(model.predict(input))




