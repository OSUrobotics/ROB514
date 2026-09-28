#!/usr/bin/env python3

import numpy as np


def convert_image(im, wall_threshold, free_threshold):
    """ Convert the image to a thresholded image with 'not seen' pixels marked
    @param im - width by height image as numpy (depends on input)
    @param wall_threshold - number between 0 and 1 to indicate wall threshold value
    @param free_threshold - number between 0 and 1 to indicate free space threshold value
    @return an image of the same WXH but with 0 (free) 255 (wall) 128 (unseen)"""

    # Assume all is unseen - fill the image with 128
    im_ret = np.zeros((im.shape[0], im.shape[1]), dtype='uint8') + 128

    im_avg = im
    if len(im.shape) == 3:
        # RGB image - convert to gray scale
        im_avg = np.mean(im, axis=2)
    # Force into 0,1
    im_avg = im_avg / np.max(im_avg)
    # threshold
    #   in our example image, black is walls, white is free
    im_ret[im_avg < wall_threshold] = 0
    im_ret[im_avg > free_threshold] = 255
    return im_ret


def open_image(im_name):
    """ A helper function to open up the image and the yaml file and threshold
    @param im_name - name of image in Data directory
    @returns image anbd thresholded image"""

    # Using imageio to read in the image
    import imageio.v2 as imageio
    im = imageio.imread("./Data/" + im_name)
   
    wall_threshold = 0.7
    free_threshold = 0.9

    im_thresh = convert_image(im, wall_threshold, free_threshold)
    return im, im_thresh


