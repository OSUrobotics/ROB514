#!/usr/bin/env python3

# This assignment uses matrices to keep track of where a robot is in the world.
#
# Note that we store geometry as lists (not numpy arrays) because json Does Not Like numpy arrays
# Slides: https://docs.google.com/presentation/d/1nTexr-lPdkq3HW4ouzYTa9iEiO-6K7j5ihHvZqixIsM/edit?usp=sharing

import numpy as np
import matplotlib.pyplot as plt
from matrix_transforms import make_scale_dict, make_translation_dict, make_rotation_dict,\
    get_dx_dy_from_matrix, get_matrix_from_sequence, get_axes_from_matrix, get_theta_from_matrix
from objects_in_world import make_blank_object, read_object, write_object, plot_object_in_own_coord_system,\
    plot_object_in_world_coord_system


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
    robot["XYs"] = robot["XYs"][0:2] + [[1, 0], [1.25, -0.5], [1, 0], [1.25, 0.5], [1, 0]] + robot["XYs"][2:]

    # Scale the square to make the robot body, then apply the desired transforms
    transform.append(make_scale_dict(2.0, 0.5))
    mat = get_matrix_from_sequence(transform)
    new_pts = mat @ robot["Pts"]

    # Convert back
    robot["XYs"] = [[xy[0], xy[1]] for xy in new_pts[1:2, :]]

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
    Should set robot["Matrix seq"] """

    # TODO: Set robot["Matrix seq"] to be a sequence of matrices that take the robot to the new position and orientation
    #  Use make_translation_dict and make_rotation_dict
# YOUR CODE HERE
    # Now build the actual matrix from the sequence
    robot["Matrix"] = get_matrix_from_sequence(robot["Matrix seq"])


def move_robot_relative(vec, d_theta):
    """ Move the robot forward by dx = vec[0], dy = vec[1], then rotate by theta
    *note* Starts from CURRENT location and orientation
    Should set robot["World location"] and robot["World orientation"]
    @param vec - the vector to move in, relative to the robot's canonical position
    @param d_theta - the amount to rotate by"""

    # TODO: Set the "Matrix seq" to be a new rotation, translation matrix by combining the current translation/rotation with the new one
    #       see get_dx_dy_from_matrix and get_theta_from_matrix and get_axes_from_matrix - you'll need these to
    #  Two ways to think about this
    #     Get the current axes (in the world) and move dx along the first axis, dy along the second
    #     Make the dx, dy vector and multiply by the current matrix to figure out what that motion is in the world
    #  They result in the same thing, mathematically
    #  Either way, the orientation is just the current orientation plus the new turn amount
# YOUR CODE HERE


# ----------------------------- plotting code -----------------------
def plot_robot_and_path_and_world(axs, world, path_to_follow, robot_path, robot):
    """ Plot the world, the path in the world, where the robot went, and where the robot is along the path
    @param - world the walls of the world
    @param - path_to_follow - the path the robot should follow
    @param - robot_path - the path the robot has taken so far
    @param - robot - the robot in its current location"""

    plot_object_in_own_coord_system(axs, world)
    plot_object_in_world_coord_system(axs, path_to_follow)
    plot_object_in_world_coord_system(axs, robot_path)
    plot_object_in_world_coord_system(axs, robot)


def plot_seq(world, path, robot, b_do_absolute):
    """Plot the robot
    @param - world the walls of the world
    @param - path - the path the robot should follow
    @param - robot - the robot
    @param - b_do_absolute - do absolute (versus relative) motion along the path"""
    n_rows = 2
    n_cols = 3
    fig, axs = plt.subplots(n_rows, n_cols)

    axs[0, 0].set_title("Robot in own coord system")
    plot_object_in_own_coord_system(axs[0, 0], robot)
    axs[0, 0].axis('equal')

    path_robot_followed = make_blank_object()
    path_robot_followed["XYs"].append([0, 0])
    path_robot_followed["Matrix"] = get_matrix_from_sequence([])

    # Move the robot to the start of the path
    move_robot_absolute(new_position=path["XYs"][0], new_orientation=0)
    for i, XYs in enumerate(path["XYs"][:-1]):
        axs_cur = axs[(i+1) // n_cols, (i+1) % n_cols]
        axs_cur.set_title(f"Step {i}")

        next_pose = path["XYs"][i+1]
        end_theta = np.arctan2(next_pose[1] - XYs[1], next_pose[0] - XYs[0])
        if b_do_absolute or i == 0:
            move_robot_absolute(new_position=path["XYs"][i], new_orientation=end_theta)
        else:
            dx = path["XYs"][i][0] - path["XYs"][i-1][0]
            dy = path["XYs"][i][1] - path["XYs"][i-1][1]
            cur_theta = get_theta_from_matrix(robot["Matrix"])
            move_robot_relative([dx, dy], end_theta - cur_theta)

        robot["Matrix"] = get_matrix_from_sequence(robot["Matrix seq"])
        new_pose = robot["Matrix"] @ np.transpose(np.array([0, 0, 1]))
        path_robot_followed["XYs"].append([new_pose[0], new_pose[1]])

        plot_robot_and_path_and_world(axs_cur, world, path, path_robot_followed, robot)
        axs_cur.axis("equal")


if __name__ == '__main__':
    make_robot("robot_centered", [])
    make_path()

    robot = read_object("robot_centered")
    world = read_object("box_centered_world")
    path = read_object("path")

    plot_seq(world, path, robot, True)

    print("Done")
