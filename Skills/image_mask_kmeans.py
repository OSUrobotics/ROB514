#!/usr/bin/env python3

# This assignment introduces you to a common task (creating a segmentation mask) and a common tool (kmeans) for
#  doing clustering of data and the difference in color spaces.

# There are no shortage of kmeans implementations out there - using scipy's
import numpy as np
from scipy.cluster.vq import kmeans, vq, whiten

# Using imageio to read in the images and skimage to do the color conversion
import imageio
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt


def read_and_cluster_image(image_name, use_hsv, n_clusters):
    """ Read in the image, cluster the pixels by color (either rgb or hsv), then
    draw the clusters as an image mask, colored by both a random color and the center
    color of the cluster
    @image_name - name of image in Data
    @use_hsv - use hsv, y/n
    @n_clusters - number of clusters (up to 6)"""

    # Read in the file
    im_orig = imageio.imread("Data/" + image_name)
    # Make sure you just have rgb (for those images with an alpha channel)
    im_orig = im_orig[:, :, 0:3]

    # The plot to put the images in
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # Make name for the image from the input parameters
    str_im_name = image_name.split('.')[0] + " "
    if use_hsv:
        str_im_name += "HSV"
    else:
        str_im_name += "RGB"

    str_im_name += f", k={n_clusters}"

    # This is how you draw an image in a matplotlib figure
    axs[0].imshow(im_orig)
    # This sets the title
    axs[0].set_title(str_im_name)

    # TODO
    # Step 1: If use_hsv is true, convert the image to hsv (see skimage rgb2hsv - skimage has a ton of these
    #  conversion routines)
    # Step 2: reshape the data to be an nx3 matrix
    #   kmeans assumes each row is a data point. So you have to give it a (widthXheight) X 3 matrix, not the image
    #   data as-is (WXHX3). See numpy reshape.
    # Step 3: Whiten the data
    # Step 4: Call kmeans with the whitened data to get out the centers
    #   Note: kmeans returns a tuple with the centers in the first part and the overall fit in the second
    # Step 5: Get the ids out using vq
    #   This also returns a tuple; the ids for each pixel are in the first part
    #   You might find the syntax data[ids == i, 0:3] = rgb_color[i] useful - this gets all the data elements
    #     with ids with value i and sets them to the color in rgb_color
    # Step 5: Create a mask image, and set the colors by rgb_color[ id for pixel ]
    # Step 6: Create a second mask image, setting the color to be the average color of the cluster
    #    Two ways to do this
    #       1) "undo" the whitening step on the returned cluster (harder)
    #       2) Calculate the means of the clusters in the original data
    #           np.mean(data[ids == c])
    #
    # Step 7: use rgb2hsv to handle the hsv option
    #   Simplest way to do this: Copy the code you did before and re-do after converting to hsv first
    #     Don't forget to take the color centers in the *original* image, not the hsv one
    #     Don't forget to rename your variables
    #   More complicated: Make a function. Most of the code is the same, except for a conversion to hsv at the beginning

    # An array of some default color values to use for making the rgb mask image
    rgb_color = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255]]
# YOUR CODE HERE
    axs[1].set_title("ID colored by rgb")
    axs[2].set_title("ID colored by cluster average")

if __name__ == '__main__':
    read_and_cluster_image("real_apple.jpg", True, 4)
    read_and_cluster_image("trees.png", True, 2)
    read_and_cluster_image("trees_depth.png", False, 3)
    read_and_cluster_image("staged_apple.png", True, 3)
    read_and_cluster_image("staged_apple.png", False, 3)
    print("done")
