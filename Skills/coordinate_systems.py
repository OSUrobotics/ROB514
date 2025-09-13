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

# -------------------- Objects ---------------------

# Global variables are Evil and Dangerous, but without using a class, this is the only sensible
#   way to do this
def onclick(event):
    global glob_current_obj
    global axs

    ix, iy = event.xdata, event.ydata
    print(f"x {ix} y {iy}")

    axs.plot(ix, iy, 'Xb')
    if glob_current_obj is None:
        glob_current_obj = make_blank_object()
        glob_current_obj["XYs"].append([ix, iy])
    else:
        glob_current_obj["XYs"].append([ix, iy])
        xs = [glob_current_obj["XYs"][-2][0], glob_current_obj["XYs"][-1][0]]
        ys = [glob_current_obj["XYs"][-2][1], glob_current_obj["XYs"][-1][1]]
        axs.plot(xs, ys, '-b')

        if np.isclose(ix, glob_current_obj["XYs"][0][0], 0.05) and np.isclose(iy, glob_current_obj["XYs"][0][1], 0.05):
            fname = input("Object name? ")
            glob_current_obj["name"] = fname
            with open("Data/" + fname + ".json", "w") as f:
                dump(glob_current_obj, f)


def make_object_by_clicking():
    """ Bring up a window. Collect (and draw) the clicks. Close the window when the user clicks
         close to the first click point"""
    global glob_current_obj
    global axs
    glob_current_obj = None
    fig, axs = plt.subplots(1, 1)
    axs.plot([-1, 1], [0, 0], '-k')
    axs.plot([0, 0], [-1, 1], '-k')

    fig.canvas.mpl_connect('button_press_event', onclick)


def get_pts(obj):
    """ Get the points out as a 3xn array, last row 1'x (i.e., homogenous points)
    @param obj - the object"""
    pts = np.ones((3, len(obj["XYs"])))
    pts[0:2, :] = np.transpose(np.array(obj["XYs"]))
    return pts

def get_matrix(obj):
    """Loop through all of the matrices, build and multiply them
    @param obj - the object"""

    obj["Matrices"] = []
    for m in obj["Matrix seq"]:
        mat_mult = np.identity(3)
        if "Identity" in m:
            mat_mult = np.identity(3)
        obj["Matrices"].append(mat_mult)

    # Now multiply them together
    mat = np.identity(3)
    for m in obj["Matrices"]:
        mat = mat @ m
    return mat


def plot_object(axs, obj, col='black'):
    """Plot the object in its own coordinate system
    @param axs - the axes of the figure to plot in
    @param obj - the object (as a dictionary)
    @param col - what color to draw it in"""
    xs = [p[0] for p in obj["XYs"]]
    ys = [p[1] for p in obj["XYs"]]
    axs.plot(xs, ys, color=col, linestyle='dashed', marker='x', label=obj["name"])


# --------------------------- Worlds ------------------
# Ok, worlds are just objects. But wall objects. Making a few different ones here, just to have some
#  different things to draw. No matrices - or rather, no scaling/rotating/translating the geometry
#  that defines the world. For this assignment.
def create_worlds():
    """ Just manually make a few worlds"""
    # This one has the "origin" of the world in the lower left corner
    box_lower_left_world = {"XYs": [[0, 0], [15, 0], [15, 10], [0, 10], [0, 0]], "name": "box_lower_left_world"}
    box_centered_world = {"XYs": [[-7, -10], [7, -10], [7, 10], [-7, 10], [-7, -10]], "name": "box_centered_world"}
    for w in [box_centered_world, box_lower_left_world]:
        with open("Data/" + w["name"] + ".json", 'w') as f:
            dump(w, f)


def plot_world(axs, world, objs, camera):
    """Plot the object in its own coordinate system
    @param axs - the axes of the figure to plot in
    @param world - the world object to plot
    @param objs - the object as a list of dictionaries"""
    xs = [p[0] for p in world["XYs"]]
    ys = [p[1] for p in world["XYs"]]
    axs.plot(xs, ys, color="blue", linestyle='solid', marker='o', label=world["name"])

    for obj in objs:
        pts = get_pts(obj)
        mat = get_matrix(obj)
        pts_in_world = mat @ pts
        axs.plot(pts_in_world[0, :], pts_in_world[1, :], '-k', label=obj["name"])


def plot_all(world, objs, camera):
    """Plot the objects and the camera in the world
    @param - world the walls of the world
    @param - objs - the objects in the world (like an arm)
    @param - camera - the camera"""
    fig, axs = plt.subplots(1, 3)

    axs[0].set_title("World")
    plot_world(axs[0], world, objs, camera)
    axs[0].axis('equal')

    axs[1].set_title("Objects")
    for obj in objs:
        plot_object(axs[1], obj)
    axs[1].legend()
    axs[1].axis('equal')

if __name__ == '__main__':
    test_matrices()

    # make_object_by_clicking()
    # create_worlds()

    names = ["box_lower_left_world", "Star"]
    objs = []
    for n in names:
        with open("Data/" + n + ".json", "r") as f:
            obj = load(f)
        objs.append(obj)
    plot_all(objs[0], objs[1:], None)

    print("Done")

