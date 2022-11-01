#!/usr/bin/env python3

# This assignment introduces you to a common task (creating a segmentation mask) and a common tool (kmeans) for
#  doing clustering of data and the difference in color spaces.
# We'll do kmeans as a stand-alone, 2D data cluster followed by using kmeans to segment an image

# There are no shortage of kmeans implementations out there - using scipy's
import numpy as np
from scipy.cluster.vq import kmeans, vq, whiten
import imageio
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv


def read_and_cluster_image(image_name, use_hsv, n_clusters):
    """ Read in the image, cluster the pixels by color (either rgb or hsv), then
    draw the clusters as an image mask, colored by both a random color and the center
    color of the cluster
    @image_name - name of image in Data
    @use_hsv - use hsv, y/n
    @n_clusters - number of clusters (up to 6)"""

    # Read in the file
    im = imageio.imread("Data/" + image_name)
    # Make sure you just have rgb (for those images with an alpha channel)
    im = im[:, :, 0:3]

    # The plot to put the images in
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    # This is how you draw an image in a matplotlib figure
    axs[0].imshow(im)

    # TODO
    # Step 1: If use_hsv is true, convert the image to hsv (see skimage rgb2hsv - skimage has a ton of these
    #  conversion routines)
    # Step 2: reshape the data to be an nx3 matrix
    #   kmeans assumes each row is a data point. So you have to give it a (widthXheight) X 3 matrix, not the image
    #   data as-is (WXHX3). See numpy reshape.
    # Step 3: Whiten the data
    # Step 4: Call kmeans with the whitened data to get out the centers
    #   Note: The centers are a 2x
    # Step 5: Get the ids out
    # Step 5: Create a mask image, and set the colors by rgb_color[ id for pixel ]
    #
# YOUR CODE HERE

if __name__ == '__main__':
    read_and_cluster_image("staged_apple.png", True, 3)
    print("done")
