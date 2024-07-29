import numpy as np
# from sklearn import  
def dataSplit(predictor_data, effector_data, percentageTraining = 0.6, shuffleProportion = 0.5):
    assert(predictor_data.shape[0] == effector_data.shape[0])
    rows = predictor_data.shape[0]
    shuffleiterations = int(rows * shuffleProportion)
    for i in range(shuffleiterations):
        u = np.random.randint(0,len(predictor_data))
        v = np.random.randint(0,len(predictor_data))
        predictor_temp = predictor_data.iloc[u]
        effector_temp = effector_data.iloc[u]

        predictor_data.iloc[u] = predictor_data.iloc[v]
        effector_data.iloc[u] = effector_data.iloc[v]

        predictor_data.iloc[v] = predictor_temp
        effector_data.iloc[v] = effector_temp
    training_predictor = predictor_data.iloc[0:int(rows * percentageTraining)]
    training_effector = effector_data.iloc[0:int(rows * percentageTraining)]
    testing_predictor = predictor_data.iloc[int(rows * percentageTraining): rows]
    testing_effector = effector_data.iloc[int(rows * percentageTraining): rows]
    return training_predictor, training_effector, testing_predictor, testing_effector

def shuffle(data, shuffleProportion = 0.5):
    rows = data.shape[0]
    shuffleiterations = int(rows * shuffleProportion)
    for i in range(shuffleiterations):
        u = np.random.randint(0,len(data))
        v = np.random.randint(0,len(data))

        temp = data.iloc[u]
        data.iloc[u] = data.iloc[v]
        data.iloc[v] = temp
    