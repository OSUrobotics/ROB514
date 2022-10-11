#!/usr/bin/env python3

# This assignment uses matrices to keep track of multiple coordinate systems (robot, world, arm, object)
#   This is all in 2D, because it's easier to draw. That includes the camera, who's "image" is just going
#    to be a line segment
#
# This should really be done with classes, not dictionaries, but that adds another layer of coding complexity.
#   So everything is dictionaries, and functions that take in dictionaries.
#
# A reminder that the "coordinate system" for the object is determined by the 2D locations of the vertices
#   used to define the object. Move all the vertices by some amount and you shift the center of the object's
#   coordinate system. One simple way to visualize a coordinate system is to draw the center point, the x and y axes,
#   and a circle (as the object). That's what the final plot does
# Topics:  Matrices, coordinate system transforms
#
# This file is JUST matrices. Do it first.

# Slides: https://docs.google.com/presentation/d/1nTexr-lPdkq3HW4ouzYTa9iEiO-6K7j5ihHvZqixIsM/edit?usp=sharing

import numpy as np
import matplotlib.pyplot as plt


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


# ---------------------------- Text versions of matrices ------------------------
def make_scale_dict(sx, sy):
    """Helper function to make a scale matrix
    @param sx scale in x
    @param sy scale in y
    @return dictionary"""
    return {"type": "scale", "sx": sx, "sy": sy}


def make_rotation_dict(theta):
    """Helper function to make a rotation matrix
    @param theta rotation
    @return dictionary"""
    return {"type": "rotate", "theta": theta}


def make_translation_dict(dx, dy):
    """Helper function to make a translation matrix
    @param dx translation in x
    @param dy translation in y
    @return dictionary"""
    return {"type": "translate", "sx": sx, "sy": sy}


def get_matrix_from_sequence(seq):
    """See slides - turn an array of dictionaries with matrix type and params into a single matrix
    Assumption: The first item in the list is the first transformation to apply to the object
    Remember: The matrix that is closest to the object vertices (the right most one) is the first applied matrix
       seq = (R, T)
        T R XYs   - R is applied first if matrices are multiplied in this order
    @param seq - list of transformations
    @return - a matrix"""

    mat = np.identity(3)
    for s in seq:
        if "translate" in s["type"]:
            next_mat = get_translation_matrix(s["dx"], s["dy"])
        elif "scale" in s["type"]:
            next_mat = get_scale_matrix(s["sx"], s["sy"])
        elif "rotate" in s["type"]:
            next_mat = get_rotation_matrix(s["theta"])
        else:
            raise ValueError(f"Expected one of translate, scale, rotate, got {s['type']}")
        # Add next matrix here (reminder: @ is matrix multiplication)
# YOUR CODE HERE
    return mat


# -------------------------------------------- Going backwards -----------
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
# YOUR CODE HERE
    axs.plot(pts_moved[0, :], pts_moved[1, :], ':g')

    axs.axis('equal')


if __name__ == '__main__':
    test_matrices()

    fig, axs = plt.subplots(2, 2)

    seq_rot_trans = [{"type":"rotate", "theta": np.pi/4.0},
                     {"type":"translate", "dx": 1, "dy": 2}]

    mat = get_matrix_from_sequence(seq_rot_trans)
    axs[0, 0].set_title("Rot trans")
    plot_axes(axs[0, 0], mat)

    # Reverse the order of operations
    seq_rot_trans.reverse()
    mat = get_matrix_from_sequence(seq_rot_trans)
    axs[0, 1].set_title("Trans rot")

    plot_axes(axs[0, 1], mat)

    seq_scl_rot_trans = [{"type":"scale", "sx":0.5, "sy":2.0},
                         {"type":"rotate", "theta": np.pi/4.0},
                         {"type":"translate", "dx": 1, "dy": 2}]

    mat = get_matrix_from_sequence(seq_scl_rot_trans)
    axs[1, 0].set_title("Scl rot trans")
    plot_axes(axs[1, 0], mat)

    # Reverse the order of operations
    seq_scl_rot_trans.reverse()
    mat = get_matrix_from_sequence(seq_scl_rot_trans)
    axs[1, 1].set_title("Trans rot scl")

    plot_axes(axs[1, 1], mat)

    print("Done")

