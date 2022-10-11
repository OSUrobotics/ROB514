#!/usr/bin/env python3

# This assignment uses matrices to keep track of multiple coordinate systems (robot, world, arm, object)
#   This is all in 2D, because it's easier to draw. That includes the camera, who's "image" is just going
#    to be a line segment
# This should really be done with classes, not dictionaries, but that adds another layer of coding complexity./
#   So everything is dictionaries, and functions that take in dictionaries.
#
# To help keep track of what is going on, I'm storing not just the final matrix, but the *sequence* of
#   matrices for each object/camera.
#
# A reminder that the "coordinate system" for the object is determined by the 2D locations of the vertices
#   used to define the object. Move all the vertices by some amount and you shift the center of the object's
#   coordinate system.
# Topics:  Matrices, coordinate system transforms

import numpy as np
import matplotlib.pyplot as plt
from json import dump, load

# -------------------- Matrices ---------------------

# These first few functions are just convenience functions for making specific transformation matrices (translation,
#  rotation, scale). Also a way to refresh your memory with the mechanics of building a specific matrix... Normally,
#  you would never write these (numpy has versions of them - in particular, use scipy's rotation/quaternion class to handle
#  rotations in 3D)
#
# A reminder that all matrices are 3x3 so that we can do translations (the upper left is the 2x2 matrix)

def get_scale_matrix(scale_x=1.0, scale_y=1.0):
    """Create a 3x3 scaling matrix
    @param scale_x - scale in x. Should NOT be 0.0
    @param scale_y - scale in y. Should NOT be 0.0
    @returns a 3x3 scaling matrix"""
    if np.isclose(scale_x, 0.0) or np.isclose(scale_y, 0.0):
        raise ValueError(f"Scale values should be non_zero {scale_x}, {scale_y}")

    mat = np.identity(3)
# YOUR CODE HERE

    return mat


def get_translation_matrix(d_x=0.0, d_y=0.0):
    """Create a 3x3 translation matrix that moves by dx, dy
    @param d_x - translate in x
    @param d_y - translate in y
    @returns a 3x3 translation matrix"""

    mat = np.identity(3)
# YOUR CODE HERE

    return mat


def get_rotation_matrix(theta=0.0):
    """Create a 3x3 rotation matrix that rotates counter clockwise by theta
    Note that it is convention to rotate counter clockwise - there's no mathematical reason for it
    @param theta - rotate by theta (theta in radians)
    @returns a 3x3 rotation matrix"""

    mat = np.identity(3)
# YOUR CODE HERE

    return mat


# These are check/debugging functions that are handy to have around. In essence, they convert from a matrix back
#   to dx, dy and theta and check for skew/non transformation matrices
# Reminder: @ is the matrix multiplication

def get_sx_sy_from_matrix(mat):
    """How does the matrix scale the x and the y axes?
    @param mat - the matrix
    @returns sx, sy - how the x and y axes are scaled"""
    diag_vec = np.zeros(shape=(3,))

    # Rather than do two vectors (one for x and one for y) it suffices to multiply the matrix
    #  by the diagonal vector (1,1,0).
    # Remember that when multiplying matrices by Vectors (as opposed to points) you should put 0 in that
    #   third coordinate, since vectors do not have a location
# YOUR CODE HERE


def get_dx_dy_from_matrix(mat):
    """Where does the matrix translate 0,0 to?
    @param mat - the matrix
    @returns dx, dy - the transformed point 0,0"""
    origin = np.zeros(shape=(3,))

    # Don't forget to turn origin into a homogenous point...
    #   Multiply the origin by the matrix then return the x and y components
    # Reminder: @ is the matrix multiplication
# YOUR CODE HERE


# Doing this one in two pieces - first, get out how the axes (1,0) and (0,1) are transformed, then in the mext
#  method get theta out of how (1,0) is transformed
def get_axes_from_matrix(mat):
    """Where does the matrix rotate (1,0) (0,1) to?
    @param mat - the matrix
    @returns x_rotated_axis, y_rotated_axis - the transformed vectors"""
    x_axis = np.zeros(shape=(3,))
    y_axis = np.zeros(shape=(3,))

    # Set x_axis to be a unit vector pointing down the x axis
    # Set y_axis to be a unit vector pointing down the y axis
    #  Multiply by the matrix to get the new "x" and "y" axes
# YOUR CODE HERE


def get_theta_from_matrix(mat):
    """ Get the actual rotation angle from how the x-axis transforms
    @param mat - the matrix
    @return theta, the rotation amount in radians"""

    # Use get_axes_from_matrix to get the x_axis, then use arctan2 to turn
    # the rotated x axis vector into an angle
    #   Use the x axis because theta for the x axis is 0 (makes the math easier)
    # Reminder: atan2 takes (y, x)
# YOUR CODE HERE
# YOUR CODE HERE
