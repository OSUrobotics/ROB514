#!/usr/bin/env python3

# How to make objects from lists of vertices and move them around (and plot them)
#
# Square: make a square and store it in a json
# Clicking: Bring up a window and let the user click points to make an object (clicking on the first point closes the
#  object)
#
# An object is just a dictionary with the vertex locations of the geometry stored in an "XYs" array.
# An object can also have a color (for drawing), a matrix sequence (see matrix_transforms.py) and the actual
#  matrix

import numpy as np
import matplotlib.pyplot as plt
from matrix_transforms import make_matrix_from_sequence
from json import dump, load


# -------------------- Objects - creating, reading, writing ---------------------
# Create a dictionary with the required keys to be an object
def make_blank_object():
    """ Matrix, pts, matrix sequence"""
    obj = {"XYs": [], "Matrix seq": [], "Matrix": None, "Color": "black", "Name": "name", "Pts": None}
    return obj


def read_object(name):
    """Read in the object from Data/name.json and convert the XYs to a numpy array
    @param name - name of the json file
    @return an object as a dictionary"""
    with open("Data/" + name + ".json", "r") as f:
        obj = load(f)
        obj["Pts"] = get_pts_as_numpy_array(obj)
        obj["Matrix"] = make_matrix_from_sequence(obj["Matrix seq"])
    return obj


def write_object(obj, name):
    """Strip out the numpy arrays before writing
    @param obj - the object
    @param name - the file name to write to"""
    obj_save_pts = obj["Pts"]
    obj_save_matrix = obj["Matrix"]

    obj["Pts"] = []
    obj["Matrix"] = []
    obj["Name"] = name

    with open("Data/" + name + ".json", 'w') as f:
        dump(obj, f)

    obj["Pts"] = obj_save_pts
    obj["Matrix"] = obj_save_matrix


# ------------------- Making a robot by clicking in screen ---------------
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


def make_square():
    """Make a square object """
    obj = make_blank_object()
    obj["XYs"] = [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]
    obj["Name"] = "Square"
    write_object(obj, "Square")


def make_wedge():
    """Make a wedge object """
    obj = make_blank_object()
    obj["XYs"] = [[-1, -1], [1, -0.8], [1, 0.8], [-1, 1], [-1, -1]]
    obj["Name"] = "Wedge"
    write_object(obj, "Wedge")


# ----------------- Some helper methods -------------------------------

def get_pts_as_numpy_array(obj):
    """ Get the points out as a 3xn array, last row 1'x (i.e., homogenous points)
    @param obj - the object
    @return the points as a 3xn numpy array """
    pts = None
    # GUIDE: Create a 3xn numpy array, with the last row 1's, and the points filled
    #  in from the obj["XYs"] key-value
    # YOUR CODE HERE
    return pts


# --------------------------- Worlds ------------------
# Ok, worlds are just objects. But wall objects. Making a few different ones here, just to have some
#  different things to draw. No matrices - or rather, no scaling/rotating/translating the geometry
#  that defines the world. For this assignment.
def create_worlds():
    """ Just manually make a few worlds"""
    # This one has the "origin" of the world in the lower left corner
    box_lower_left_world = make_blank_object()
    box_lower_left_world["XYs"] =  [[0, 0], [15, 0], [15, 10], [0, 10], [0, 0]]
    box_lower_left_world["Color"] = "darkgrey"
    write_object(box_lower_left_world, "Box_lower_left_world")

    box_centered_world = make_blank_object()
    box_centered_world["XYs"] = [[-7, -10], [7, -10], [7, 10], [-7, 10], [-7, -10]]
    box_centered_world["Color"] = "darkgrey"
    write_object(box_centered_world, "Box_centered_world")


# --------------------------- Plotting ------------------
def plot_object_in_own_coord_system(axs, obj):
    """Plot the object in its own coordinate system
    @param axs - the axes of the figure to plot in
    @param obj - the object (as a dictionary)"""
    xs = [p[0] for p in obj["XYs"]]
    ys = [p[1] for p in obj["XYs"]]
    col = 'black'
    if "Color" in obj:
        col = obj["Color"]

    axs.plot(xs, ys, color=col, linestyle='dashed', marker='x', label=obj["Name"])


def plot_object_in_world_coord_system(axs, obj, in_matrix=None):
    """Plot the object in the world by applying in_matrix (if specified) followed by the
      matrix transformation already in the object
    @param axs - the axes of the figure to plot in
    @param obj - the object (as a dictionary)
    @param matrix - an additional matrix to multiply the geometry by"""

    matrix = in_matrix
    if in_matrix is None:
        matrix = np.identity(3)

    # This only checks if the numpy array is the same size, not that the values are the same
    try:
        if len(obj["XYs"]) != obj["Pts"].shape[1]:
            obj["Pts"] = get_pts_as_numpy_array(obj)
    except:
        obj["Pts"] = get_pts_as_numpy_array(obj)

    # This multiplies the matrix by the points
    try:
        pts_in_world = matrix @ obj["Matrix"] @ obj["Pts"]
    except ValueError or KeyError:
        obj["Matrix"] = make_matrix_from_sequence(obj["Matrix seq"])
        pts_in_world = matrix @ obj["Matrix"] @ obj["Pts"]

    col = 'black'
    if "Color" in obj:
        col = obj["Color"]

    axs.plot(pts_in_world[0, :], pts_in_world[1, :], color=col, linestyle='solid', label=obj["Name"])


def plot_world(axs, world, objs, camera):
    """Plot the object in its own coordinate system
    @param axs - the axes of the figure to plot in
    @param world - the world object to plot
    @param objs - the object as a list of dictionaries"""
    xs = [p[0] for p in world["XYs"]]
    ys = [p[1] for p in world["XYs"]]
    axs.plot(xs, ys, color="blue", linestyle='solid', marker='o', label=world["Name"])

    for obj in objs:
        plot_object_in_world_coord_system(axs, obj)


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
        plot_object_in_world_coord_system(axs[1], obj)
    axs[1].legend()
    axs[1].axis('equal')


if __name__ == '__main__':
    # make_square()
    make_wedge()

    # make_object_by_clicking()
    create_worlds()

    names = ["box_lower_left_world", "Star"]
    objs = []
    for n in names:
        with open("Data/" + n + ".json", "r") as f:
            obj = load(f)
        objs.append(obj)
    plot_all(objs[0], objs[1:], None)

    # Depending on if your mac, windows, linux, and if interactive is true, you may need to call this to get the plt
    # windows to show
    plt.show()
    print("Done")

