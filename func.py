import numpy as np


def gaussian(x, t):
    return np.exp(-np.square(x * t))


def gaussian_second_derivative(x, t):
    return (2 - 4 * np.square(x)) * np.exp(-np.square(x*t))

def euclidean_distance(x, y):
    if len(x.shape) == 3:
        return np.linalg.norm(x-y, axis=2)#np.sum(np.square(x - y), axis=2)
    else:
        return np.linalg.norm(x-y) #np.sum(np.square(x - y))


def rectangular_distance(n1, n2):
    return np.abs(n1[0] - n2[0]) + np.abs(n1[1] - n2[1])

