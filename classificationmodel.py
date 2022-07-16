import numpy as np
import pickle

loaded_pickle_model = pickle.load(open("classifier_model.pkl", "rb"))


def deepneuralnetwork(int_features):
    array_feature = np.array(int_features)
    dimention_feature = np.expand_dims(array_feature, axis=0)
    output = loaded_pickle_model.predict(dimention_feature)
    if output == 1:
        return " Presence of cardiovascular disease"
    else:
        return " Absence of cardiovascular disease"


