#%%

import numpy as np

np.set_printoptions(suppress=True)

def softmax(w):
    e = np.exp(np.array(w))
    dist = e / np.sum(e)
    return dist

def sigmoid(x):
    return 1/ (1 + np.exp(-np.array(x)))

