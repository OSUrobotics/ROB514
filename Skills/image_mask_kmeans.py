#!/usr/bin/env python3

# This assignment introduces you to a common task (creating a segmentation mask) and a common tool (kmeans) for
#  doing clustering of data and the difference in color spaces.
# We'll do kmeans as a stand-alone, 2D data cluster followed by using kmeans to segment an image

# There are no shortage of kmeans implementations out there - using scipy's
import numpy as np
from scipy.cluster.vq import kmeans, vq, whiten
from numpy.random import normal, uniform, randint
import matplotlib.pyplot as plt


def make_n_clusters(n_clusters, n_data_pts):
    """ Use a normal distribution to create a bunch of samples around a few clusters (all 2d)
     @param n_clusters - how many cluster centers to use
     @param n_data_pts - how many data points, total, to generate
     @returns cluster centers, data points, center data point came from, as numpy arrays"""

    centers = uniform(0, 1, (n_clusters, 2))
    data = np.zeros((n_data_pts, 2))
    ids = np.zeros(n_data_pts, dtype=int)
    for i in range(0, n_data_pts):
        # Randomly pick a cluster
        ids[i] = randint(0, n_clusters)
        data[i, :] = normal(centers[ids[i]], [0.2, 0.175])

    return centers, data, ids


def plot_clusters_and_data(axs, centers, data, ids):
    """ Plot the data so we can check that it looks ok
    @param centers - the center locations of the data
    @param data - the actual data"""
    cols = ["salmon", "green", "goldenrod", "magenta", "green", "grey"]
    cols_x = ["darksalmon", "darkgreen", "darkgoldenrod", "darkmagenta", "darkgreen", "darkgrey"]
    for i in range(0, centers.shape[0]):
        axs.scatter(data[ids == i, 0], data[ids == i, 1], marker='.', color=cols[i])

    for i in range(0, centers.shape[0]):
        axs.plot(centers[i][0], centers[i][1], marker='X', color="black", markersize='6')
        axs.plot(centers[i][0], centers[i][1], marker='X', color=cols_x[i], markersize='5')
    axs.axis("equal")


def find_cluster_centers(data, n_clusters):
    """ Find the centers and assign IDs to the points
    @param data - the data as a numpy array
    @returns centers, ids for each data point"""
    data_normalized = whiten(data)
    centers = kmeans(data_normalized, n_clusters)
    ids = vq(data_normalized, centers[0])




def run_2d_test():
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    centers, data, ids = make_n_clusters(4, 1000)
    plot_clusters_and_data(axs[0], centers, data, ids)

if __name__ == '__main__':
    run_2d_test()
    print("done")
