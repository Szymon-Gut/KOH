import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from func import gaussian, euclidean_distance, gaussian_second_derivative, rectangular_distance
from visualisation_utils import *
from sklearn.metrics import adjusted_rand_score


class Network:
    def __init__(self, input_shape, shape, metric=euclidean_distance,
                 neighbourhood_func=gaussian, min_val=-1, max_val=1):
        self.all_data = None
        self.neighbourhood_scale = None
        self.data = None
        self.learning_rate = None
        self.n_epochs = None
        self.metric = metric
        self.neighbourhood_func = neighbourhood_func
        self.input_shape = input_shape
        self.shape = shape
        self.weights = None
        self.neurons = None
        self._initialize_weights(min_val, max_val)

    def _initialize_weights(self, min_val, max_val):
        self.weights = np.random.uniform(min_val, max_val, size=[self.shape[0],
                                                                 self.shape[1],
                                                                 self.input_shape])

    def decay_function(self, t):
        return np.exp(-t / self.n_epochs)

    def distance_between_neighbors(self, n1, n2):
        return np.abs(n1[0] - n2[0]) + np.abs(n1[1] - n2[1])

    def neighbourhood_weights(self, n1, n2, t):
        distance = self.distance_between_neighbors(n1, n2)
        return self.neighbourhood_func(distance * self.neighbourhood_scale, t)

    def _find_BMU(self, x):
        distances = self.metric(self.weights, x)
        return np.unravel_index(np.argmin(distances, axis=None),
                                distances.shape)


    def fit(self, all_data, n_epochs, neighbourhood_scale, print_results = True):
        self.n_epochs = n_epochs
        self.neighbourhood_scale = neighbourhood_scale
        self.all_data = all_data
        self.data = all_data[:, np.arange(self.input_shape)]

        for t in range(n_epochs):
            np.random.shuffle(self.data)
            for x in self.data:
                BMU_coord = self._find_BMU(x)
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        n1 = np.array(BMU_coord)
                        n2 = np.array([i, j])
                        nw = self.neighbourhood_weights(n1, n2, t)
                        delta_weights = nw * self.decay_function(t) * (x - self.weights[i, j, :])
                        self.weights[i, j, :] += delta_weights
            if print_results:
                if t % 6 == 1:
                    self.plot()
                evaluation = self.check_clustering(all_data[:, np.arange(self.input_shape)],all_data[:,-1])
                print(f'Epoch no. {t}, wartość adjusted_rand_score: {evaluation}')
        if print_results:
            print("\n\nPODSUMOWANIE")
            evaluation = self.check_clustering(all_data[:, np.arange(self.input_shape)],all_data[:,-1])
            print(f"Uzyskana wartość adjusted_rand_score: {evaluation}")
            print(f"Używana funkcja sąsiedztwa: {self.neighbourhood_func.__name__}")
            plot_predicted_and_true_clusters(self, all_data)

    def assign_data_to_clusters(self, data):
        centers = np.reshape(self.weights, (-1, self.input_shape))
        data_clusters = []
        for row in data:
            distances = np.array(
                [euclidean_distance(row, center) for center in centers])
            data_clusters.append(np.argmin(distances))
        return np.array(data_clusters)
    
    def plot(self):
        if self.input_shape == 2:
            plot_2D_clustered(self)

        if self.input_shape == 3:
            plot_3D_clustered(self)

    def check_clustering(self, x, y):
        return adjusted_rand_score(y, self.assign_data_to_clusters(x))
