#!/usr/bin/env python3

# This assignment uses matrices to keep track of where a robot is in the world.
#
# Note that we store geometry as lists (not numpy arrays) because json Does Not Like numpy arrays
# Slides: https://docs.google.com/presentation/d/1nTexr-lPdkq3HW4ouzYTa9iEiO-6K7j5ihHvZqixIsM/edit?usp=sharing

import numpy as np
import matplotlib.pyplot as plt
from matrix_transforms import make_scale_dict, make_translation_dict, make_rotation_dict,\
    get_dx_dy_from_matrix, get_matrix_from_sequence, get_axes_from_matrix
from objects_in_world import make_blank_object, read_object, write_object


# -------------------------- Making robot geometry ---------------------------
# You don't have to edit these - I'm just putting them in here so you can see how I made the robot and the robots
#  with different coordinate systems
def make_robot(name, transform):
    """ Make a robot from a square and a wedge for a camera (the front of the robot)
    This method makes the robot longer than it is wide, centers it at 0,0, pointed down the x-axis with the
    'camera' on the front
    @param name for the robot
    @param transform to apply to the robot body to change its center, orientation"""

    # Has all of the information needed to draw an object
    robot = read_object("Square")

    robot["Color"] = "indigo"
    robot["Name"] = "Robot"

    # Add the wedge to the existing box to be the camera
    robot["XYs"].insert([[1, 0], [1.25, -0.5], [1, 0], [1.25, 0.5], [1, 0]], 2)

    # Scale the square to make the robot body, then apply the desired transforms
    seq = transform.append(make_scale_dict(2.0, 0.5))
    mat = get_matrix_from_sequence(seq)
    new_pts = mat @ robot["Pts"]

    # Convert back
    robot["XYs"] = [[xy[0], xy[1]] for xy in new_pts[1:2, :]]

    # Robot's current position and orientation in the world
    robot["World location"] = [0, 0]
    robot["World orientation"] = 0

    write_object(robot, name)


def make_path():
    """ a path for the robot to follow (in this case, just an l-shape starting at the origin
    @return a dictionary with two objects (the camera and the body of the robot) and an initial translation/rotation
    (set to zero) """

    # Has all of the information needed to draw an object
    path = make_blank_object()

    path["Color"] = "grey"
    path["Name"] = "Path"

    # Add the wedge to the existing box to be the camera
    path["XYs"] = [[0, 0], [0.0, -0.5], [0.5, -0.5], [0.5, 0.5]]

    write_object(path, "Path")


# ----------------------------- move the robot -----------------------
def move_robot_absolute(new_position, new_orientation):
    """ Assign the robot a new position and orientation in the world
    Should set robot["World location"] and robot["World orientation"]"""

# YOUR CODE HERE


def move_robot_relative(dx_dy, d_theta):
    """ Move the robot forward by dx, dy, then rotate by theta
    *note* Starts from CURRENT location and orientation
    Should set robot["World location"] and robot["World orientation"]"""

# YOUR CODE HERE


# ----------------------------- plotting code -----------------------
def plot_robot_in_world(axs, robot):
    """ Move the robot's center to world location, and orient by world orientation"""

    # You must use this format - create a matrix to take the robot to the world, then apply that to
    # the robot's pts

# YOUR CODE HERE
