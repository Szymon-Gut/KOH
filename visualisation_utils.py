import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score


def plot_true(data):

    if data.shape[-1] == 3:
        plt.scatter(x=data[:, 0], y=data[:, 1], c=data[:, 2])
    elif data.shape[-1] == 4:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=data[:, 3])
    plt.title('Prawdziwy podział na klastry')


def plot_predicted_and_true_clusters(network, true_data):
    if true_data.shape[-1] == 3:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plot_true(true_data)
        plt.subplot(1, 2, 2)
        plot_2D_clustered(network)

    elif true_data.shape[-1] == 4:
        plot_true(true_data)
        plot_3D_clustered(network)
        plt.show()


def plot_2D_not_clustered(weights):
    plot_weights_2D(weights)
    plt.title('Pozycje neuronów')
    plt.show()


def plot_2D_clustered(network):
    plt.scatter(network.data[:, 0], network.data[:, 1], c=network.assign_data_to_clusters(network.data))
    plot_weights_2D(network.weights)
    plt.title('Podział na klastry według sieci Kohonena')
    plt.xlim(-4, 4)
    plt.ylim(-10, 10)
    plt.show()


def plot_weights_2D(weights):
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            if len(weights.shape) == 3:
                w = weights[i, j]
                plt.scatter(x=[w[0]], y=[w[1]], color='black', s=50)


def plot_weights_3D(weights, ax):
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            if len(weights.shape) == 3:
                w = weights[i, j]
                ax.scatter([w[0]], [w[1]], [w[2]], color='black', s=50)


def plot_3D_clustered(network):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(network.data[:, 0], network.data[:, 1], network.data[:, 2], c=network.assign_data_to_clusters(network.data))
    plot_weights_3D(network.weights, ax)
    plt.title('Podział na klastry według sieci Kohonena')
    plt.show()



