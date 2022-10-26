#!/usr/bin/env python3

# The usual imports
import numpy as np
import matplotlib.pyplot as plt

# --------------------------- Forward IK ------------------
# The goal of this part of the assignment is to use matrices to position a robot arm in space.
#
# Slides: https://docs.google.com/presentation/d/11gInwdUCRLz5pAwkYoHR4nzn5McAqfdktITMUe32-pM/edit?usp=sharing
#

import matrix_transforms as mt
import objects_in_world as obj


# ------------------ Step 1: Object transform matrices
# Use matrices to take two basic shapes (a square and a wedge) and re-shape them into the geometry for the arm.
#
# Yes, you could just create these with the "correct" XYs, but use matrices because that's what most packages
#  actually do - they create the geometry (in whatever package) and then in the URDF they might add additional
#  transforms to scale, position, and orient them the way they want
#
# For all of these, you should be doing some version of
#    obj["Matrix_seq"].append(mt.make_xxx_dict(params))
# where xxx is a scale followed by a rotate (maybe) followed by a translate
# See slides for what the resulting shapes look like
def set_transform_base(obj_wedge):
    """ Position and orient the base of the object
    Base middle should be at 0,0, wedge pointed up, 1 unit wide, 0.5 units tall
    @param obj_wedge - the wedge object to make the arm base out of"""

    # BEGIN SOLUTION
    obj_wedge["Matrix_seq"].append(mt.make_scale_dict(0.5, 0.25))
    obj_wedge["Matrix_seq"].append(mt.make_rotate_dict(np.pi/2))
    obj_wedge["Matrix_seq"].append(mt.make_translate_dict(0.0, 0.25))
    # END SOLUTION


def create_arm_geometry():
    """ Read in the square/wedge matrices, then call the set_transform_* functions to move them around
    @returns A list of the objects
    """

    base = obj.read_object("Wedge")
    set_transform_base(base)

    return [base]

