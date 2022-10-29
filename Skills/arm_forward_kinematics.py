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
from objects_in_world import read_object, plot_object_in_world_coord_system


# ------------------ Step 1: Object transform matrices -------------------------------------
# Use matrices to take two basic shapes (a square and a wedge) and re-shape them into the geometry for the arm, gripper.
#
# Yes, you could just create these with the "correct" XYs, but use matrices because that's what most packages
#  actually do - they create the geometry (in whatever package) and then in the URDF they might add additional
#  transforms to scale, position, and orient them the way they want
#
# For all of these, you should be doing some version of
#    obj["Matrix seq"].append(mt.make_xxx_dict(params))
# where xxx is a scale followed by a rotate (maybe) followed by a translate
# See slides for what the resulting shapes look like
# Wedge and Square are both -1,-1 to 1, 1
def set_transform_base(obj_wedge, base_width=1.0, base_height=0.5):
    """ Position and orient the base of the object
    Base middle should be at 0,0, wedge pointed up, base_width wide, base_height tall
    @param obj_wedge - the wedge object to make the arm base out of
    @return the modified object"""

    # TODO: append transformations to obj_wedge["Matrix seq"] to get it in the right position/size/orientation
    #  Scale first, then rotate, then translate
    # Note that the part where we actually multiply the points (currently stored in the key "XYs") by the matrix
    #  (stored in the key "Matrix") will happen when we make the plot
    # Open up Data/Wedge.json if you want to see the XYs (this shape is made in objects_in_world.py)
# YOUR CODE HERE
    # Force recalculation of matrix
    obj_wedge["Matrix"] = mt.make_matrix_from_sequence(obj_wedge["Matrix seq"])

    # Setting up for being able to position and orient the arm somewhere else
    obj_wedge["Angle"] = 0.0
    obj_wedge["Position"] = 0.0
    obj_wedge["Color"] = "darkturquoise"
    return obj_wedge


def set_transform_link(obj_square, arm_length, arm_height):
    """ This is one of the arm components - since they're all kinda the same (just different sizes) just have
    one function to create them
    The arm should have the left hand side, middle at 0,0 and extend along the x_axis by length
    @param obj_square - the square
    @param arm_length - the desired length of the arm
    @param arm_height - the desired height of the arm
    @return the modified object"""

    # TODO: append transformations to obj_square["Matrix seq"] to get it in the right position/size/orientation
    #  Reminder that squares are defined by -1,-1 to 1,1, and so currently have side lengths of 2...
# YOUR CODE HERE

    # Force recalculation of matrix
    obj_square["Matrix"] = mt.make_matrix_from_sequence(obj_square["Matrix seq"])

    # Save the arm length in the dictionary
    obj_square["Arm length"] = arm_length
    obj_square["Angle"] = 0
    return obj_square


def set_transform_palm(obj_square, palm_width):
    """ This is palm of the gripper - a rectangle palm_width tall, centered at the origin, 1/10 as wide as it is tall
    @param obj_square - square shape to start with
    @param palm_width - the desired separation of the two fingers
    @return the modified object"""

    # TODO: append transformations to obj_square["Matrix seq"] to get it in the right position/size/orientation
# YOUR CODE HERE

    # Force recalculation of matrix
    obj_square["Matrix"] = mt.make_matrix_from_sequence(obj_square["Matrix seq"])

    # Keep this for later
    obj_square["Palm width"] = palm_width
    obj_square["Angle"] = 0  # Wrist angle
    obj_square["Color"] = "tomato"
    obj_square["Name"] = "Palm"
    return obj_square


def set_transform_finger(obj_wedge, palm_width, finger_size, b_is_top):
    """ This is one of the fingers - two wedges, separated by the palm width
    @param obj_wedge - wedge shape to start with
    @param palm_width - the desired separation of the two fingers
    @param finger_size - tuple, how long and wide to make the finger
    @param b_is_top - is this the top or the bottom finger?
    @return the modified object"""

    # TODO: append transformations to obj_wedge["Matrix seq"] to get it in the right position/size/orientation
    #  b_is_top means it's the top finger...
# YOUR CODE HERE

    # Force recalculation of matrix
    obj_wedge["Matrix"] = mt.make_matrix_from_sequence(obj_wedge["Matrix seq"])

    # Which finger?
    if b_is_top:
        obj_wedge["Location"] = "Top"
        obj_wedge["Color"] = "darkgoldenrod"
    else:
        obj_wedge["Location"] = "Bottom"
        obj_wedge["Color"] = "darkgreen"

    # Keep this for later
    obj_wedge["Palm width"] = palm_width
    obj_wedge["Angle"] = 0
    return obj_wedge


def create_gripper(palm_width, finger_size):
    """ Make a gripper from a palm and two fingers
    @param palm_width - the desired separation of the two fingers
    @param finger_size - how long and wide to make the finger
    @return the modified object"""

    obj_square = read_object("Square")
    palm = set_transform_palm(obj_square, palm_width)
    palm["Grasp"] = finger_size[0] * 0.75
    gripper = [palm]
    for b in [True, False]:
        obj_wedge = read_object("Wedge")
        finger = set_transform_finger(obj_wedge, palm_width, finger_size, b)
        finger["Name"] = f"Finger {finger['Location']}"
        finger["Angle"] = 0.0
        gripper.append(finger)

    return gripper


def create_arm_geometry(base_size, link_sizes, palm_width, finger_size):
    """ Read in the square/wedge matrices, then call the set_transform_* functions to move them around
    See slides for what these should look like when done
    @param link_sizes: A list of tuples, one for each link, with the link length and width
    @param palm_width: The width of the palm (how far the fingers are spaced apart)
    @param finger_size: A tuple with the finger length and width
    @returns A list of the objects (arm_geometry)
    """

    # This is the base of the robot
    base = read_object("Wedge")
    set_transform_base(base, base_size[0], base_size[1])
    base["Name"] = "Base"

    # Keep the geometry in a list - start the list with the base object
    arm_geometry = [base]

    # Since arm links are kinda the same, just different sizes, create a list with the link sizes and make one
    #  component from each pair of sizes
    # Note that this makes it easy to add (or remove) links by just changing this list
    for i, size in enumerate(link_sizes):
        square = read_object("Square")
        # Note that, because dictionaries are pointers and we edited square inside of set_transform, square and arm
        #   link point to the same thing after this call - which is why we'll read in a new copy of square on the
        #   next loop
        arm_link = set_transform_link(square, size[0], size[1])
        arm_link["Name"] = f"Link {i}"

        # Add this version to our list of arm components
        arm_geometry.append(arm_link)

    # The gripper is made of Three pieces - we're going to put them all together (as a list) at the end of the list...
    #   This will make it possible to build the matrix for the gripper and apply that matrix to all gripper elements
    gripper = create_gripper(palm_width, finger_size)

    # Add the gripper as a single element to the end of the list
    arm_geometry.append(gripper)
    return arm_geometry


# ----------------- Adding angles --------------------------
def set_angles_of_arm_geometry(arm_geometry, angles):
    """ Just sets the angle of each link, gripper
    Assumes that the arm has as many links as the angles and that the gripper is two objects at the end.
    There are three numbers for that last angle - a wrist angle, and one angle for each finger, stored as a list
    Not setting the base transformation
    @param arm_geometry - the arm geometry
    @param angles - the actual angles"""

    if len(arm_geometry) != len(angles) + 1:
        raise ValueError("Arm geometry and angles do not match")

    # Set angles of joints
    for comp, ang in zip(arm_geometry[1:-1], angles[0:-1]):
        comp["Angle"] = ang

    # Set angle of wrist and fingers
    #   Note that these are all dictionaries, so this is just a pointer to the last list element
    gripper = arm_geometry[-1]
    gripper[0]["Angle"] = angles[-1][0]
    for i, comp in enumerate(gripper):
        comp["Angle"] = angles[-1][i]


def get_matrix_base(base_link):
    """ Get a matrix that will move the first link to the top of the base, pointed up
    This should work even if we change the orientation, location, and scale of the base
    @param base_link is the object that is the base
    @return 3x3 matrix that takes the origin, translates it to the top of the base, and makes x point up"""

    # TODO:
    #  Figure out where (1.0, 0.0) went on the base wedge (that's the translation you need to apply)
    #  Figure out how (1, 0) should be rotated to make it point up
    #    Reminder: mt.get_xx_from_matrix is helpful here...
    #    Rotate first, then translate
# YOUR CODE HERE


def get_rotation_link(arm_link):
    """ Get JUST the rotation matrix for this link
    @param arm_link - the link dictionary, angle stored in arm_link['Angle']
    @return 3x3 rotation matrix"""

    # TODO Create a rotation matrix based on the link's angle (stored with the key "Angle")
# YOUR CODE HERE


def get_matrix_link(arm_link):
    """ Get a matrix that will move the next link to the end of this one, pointed along the x axis, then rotate
    both this link AND the next one by the angle of this link
    This should work even if we change the scale of the link
    @param arm_link is the object that is the arm link
    @return 3x3 matrix that takes the origin, translates it to the end of the link, and then rotates by angle"""

    # TODO:
    #  Figure out where (1.0, 0.0) went on the link (that's the translation you need to apply)
    #  Figure out how (1,0) should be rotated to make it point up
    #    Reminder: mt.get_xx_from_matrix is helpful here...
    #    Rotate first, then translate
# YOUR CODE HERE


def get_matrices_all_links(arm_with_angles):
    """ Get a list of matrices (one for each link, plus one for the base and one for the gripper) that we can
    use to move each component to its final location
    First matrix: The matrix to multiply the base by (should be the identity)
    Second matrix: The matrix to multiply the first link by in order to attach it to the base, pointing up
    Third matrix: The matrix to multiply the second link by in order to attach it to the first link, pointing
       in the same direction as the first link
       ...
    Last matrix: The matrix to multiply the gripper by in order to attach it to the last link, pointing in the
    same direction as the last link"""

    # First matrix is the one for the base link - should be the identity
    matrices = [np.identity(3)]

    # Matrix for the first link - the rotation plus translation to put the arm on top of the base link
    matrices.append(get_matrix_base(arm_with_angles[0]))

    # Now do all of the links - the last matrix is the one that is applied to the gripper
    for link in arm_with_angles[1:-1]:
        # TODO: append a matrix to the list that is the matrix that we will multiply this link from
        #   In other words, multiply the last matrix by the matrix for this link then add it to the list
# YOUR CODE HERE

    return matrices


def get_matrix_finger(finger):
    """ Get the matrix that moves the finger to the correct angle
    Pulling this out as a method so we can use it elsewhere
    @param finger - the finger as a dictionary
    @return a 3x3 matrix"""

    # TODO:
    #   Translate the base of the finger back to the origin, rotate it, then translate it back out
    #   Reminder: The base of the finger can be found using mt.get_dx_dy_from_matrix
# YOUR CODE HERE
    return matrix


# ----------------- Gripper location --------------------------
def get_gripper_location(arm_with_angles):
    """ Get the gripper grasp location (between the fingers) given the arm
    Assumes the angles are stored in the arm links.
    Use the "Grasp" key in the gripper to figure out how far to offset the point point from the base of the gripper
    @param arm_with_angles
    @return x,y as a tuple - the location of the "grasp" point in the gripper
    """
    gripper = arm_with_angles[-1]
    grasp_dist = gripper[0]["Grasp"]  # The distance out along the x axis that we'll call the "grasp" point

    # TODO:
    # Step 1: Get the matrices
    # Step 2: Use the last matrix plus the rotation of the wrist to build a matrix for the gripper
    # Step 3: Multiply the last matrix by [d, 0] to get the location in world coordinates
# YOUR CODE HERE
    # Format for returning a tuple
    return (0, 0)


def get_gripper_orientation(arm_with_angles):
    """ Get a vector pointing out of the palm from the arm with angles
    Assumes the angles are stored in the arm links.
    @param arm_with_angles
    @return vx, vy as a tuple - the vector out of the grasp (unit length)
    """
    gripper = arm_with_angles[-1]

    # TODO:
    # Step 1: Get the matrices
    # Step 2: Use the last matrix plus the rotation of the wrist to build a matrix for the gripper
    # Step 3: Get the matrix that takes (1,0) to the world
# YOUR CODE HERE
    # Format for returning a tuple
    return (1, 0)

# ----------------- Plotting routines --------------------------

def plot_arm_components(axs, arm, b_with_angles=False):
    """ Plot each component in its own coordinate system with the matrix that was used to chagne the geometry
    @param axs are the axes of each plot window
    @param arm as a list of dictionaries, each of which is an object
    @param b_with_angles - if True, rotate the geometry by the given angle stored in the object """

    box = read_object("Square")
    box["Matrix seq"].append(mt.make_scale_dict(0.75, 0.75))
    box["Color"] = "lightgrey"
    box["Matrix"] = mt.make_matrix_from_sequence(box["Matrix seq"])

    # if b_with_angles is false, this stays the identity matrix for all plots
    rot_matrix = np.identity(3)

    # Plot the base and the arm links in their own windows
    for i, component in enumerate(arm[:-1]):
        plot_object_in_world_coord_system(axs[i], box)

        # TODO: if b_with_angles is True, set the rotation matrix by the angle stored in the component
        #   Reminder: mt.make_rotation_matrix will make a rotation matrix
        #   Reminder: the angle is stored with the key "Angle"
        if b_with_angles:
            rot_matrix = get_rotation_link(component)

        plot_object_in_world_coord_system(axs[i], component, rot_matrix)
        axs[i].set_title(component["Name"])
        axs[i].plot(0, 0, 'xr')
        axs[i].axis("equal")

    # plot the gripper - I know it's more than one piece of geometry, so plot each in turn
    plot_object_in_world_coord_system(axs[-1], box)

    gripper = arm[-1]

    # Palm first
    palm = gripper[0]
    if b_with_angles:
        rot_matrix = get_rotation_link(palm)

    plot_object_in_world_coord_system(axs[-1], palm, rot_matrix)

    # Now the fingers
    for finger in gripper[1:]:
        # TODO: Rotate each finger by the given amount, then rotate the entire gripper by the wrist angle
        # Step 1: Edit get_matrix_finger to get the matrix to move just the finger
        # Step 2: Multiply that matrix by the rotation matrix for the palm
# YOUR CODE HERE
        plot_object_in_world_coord_system(axs[-1], finger, rot_matrix)

    # Draw a red line for the palm and an x at the base of the wrist and another at the finger contact points

    # Get the y axis of the rotation matrix
    axs_x, axs_y = mt.get_axes_from_matrix(mt.make_rotation_matrix(palm["Angle"]))
    # Scale by palm width
    axs_y = axs_y * palm["Palm width"] / 2.0
    # Go from -vec to + vec
    axs[-1].plot([-axs_y[0], axs_y[0]], [-axs_y[1], axs_y[1]], '-r')

    # Draw a red + at the base of the palm
    axs[-1].plot(0, 0, '+r')

    # Draw a green x at the contact point of the fingers
    axs[-1].plot(axs_x[0] * palm["Grasp"], axs_x[1] * palm["Grasp"], 'xg')

    axs[-1].set_title("Gripper")
    axs[-1].legend(loc="upper left")
    axs[-1].axis("equal")


def plot_complete_arm(axs, arm, matrices):
    """ From left to right, work backwards, plotting first the gripper, then each link in turn, and finally, the
    whole thing on the base
    @param axs are the axes of each plot window
    @param arm as a list of dictionaries, each of which is an object
    @param matrices - the matrices for each component (the TRTRTR) """

    box = read_object("Square")
    box["Matrix seq"].append(mt.make_scale_dict(2.0, 2.0))
    box["Color"] = "lightgrey"
    box["Matrix"] = mt.make_matrix_from_sequence(box["Matrix seq"])

    # From left to right,
    plot_object_in_world_coord_system(axs, box)
    origin = np.transpose(np.array([0, 0, 1]))
    for i, component in enumerate(arm[:-1]):
        new_origin = matrices[i] @ origin
        axs.plot(new_origin[0], new_origin[1], 'g+')
        plot_object_in_world_coord_system(axs, component, matrices[i] @ get_rotation_link(component))

    gripper = arm[-1]
    # The palm
    wrist_rotation = get_rotation_link(gripper[0])
    plot_object_in_world_coord_system(axs, gripper[0], matrices[-1] @ wrist_rotation)

    for finger in gripper[1:3]:
        plot_object_in_world_coord_system(axs, finger, matrices[-1] @ wrist_rotation @ get_matrix_finger(finger))

    # These will work when you do step 4
    x, y = get_gripper_location(arm)
    axs.plot(x, y, '+g', label="Grasp location")

    vx, vy = get_gripper_orientation(arm)
    axs.plot([x, x + vx * gripper[0]["Grasp"]], [y, y + vy * gripper[0]["Grasp"]], '-g', label="Grasp orientation")

    axs.set_title("Arm")
    axs.axis("equal")
    axs.legend(loc="lower left")


if __name__ == '__main__':
    # Step 1 - create the arm geometry
    base_size_param = (1.0, 0.5)
    link_sizes_param = [(0.5, 0.25), (0.3, 0.1), (0.2, 0.05)]
    palm_width_param = 0.1
    finger_size_param = (0.075, 0.025)

    # This function calls each of the set_transform_xxx functions, and puts the results
    # in a list (the gripper - the last element - is a list)
    arm_geometry = create_arm_geometry(base_size_param, link_sizes_param, palm_width_param, finger_size_param)
    if len(arm_geometry) != 5:
        print("Wrong number of components, should be 5, got {len(arm_geometry)}")
    if len(arm_geometry[-1]) != 3:
        print("Wrong number of gripper components, should be 3, got {len(arm_geometry[-1])}")

    # Should show all 5 components, the base, 3 links, and the gripper
    # Step 1 - note, comment out this one if you don't want both drawn on top of each other when you do step 2
    fig, axs = plt.subplots(1, len(arm_geometry), figsize=(4 * len(arm_geometry), 4))
    plot_arm_components(axs, arm_geometry)

    # Step 2 - rotate each link element in its own cooridinate system
    # Several different angles to check your results with
    angles_none = [0.0, 0.0, 0.0, [0.0, 0.0, 0.0]]
    angles_check_fingers = [np.pi/2, -np.pi/4, -3.0 * np.pi/4, [0.0, np.pi/4.0, -np.pi/4.0]]
    angles_check_wrist = [np.pi/2, -np.pi/4, -3.0 * np.pi/4, [np.pi/3.0, 0.0, 0.0]]
    angles_check = [np.pi/2, -np.pi/4, -3.0 * np.pi/4, [np.pi/3.0, np.pi/4.0, -np.pi/4.0]]
    set_angles_of_arm_geometry(arm_geometry, angles_check)
    # plot_arm_components(axs, arm_geometry, b_with_angles=True)

    # Step 3 & 4 - step 4 adds in drawing the green + for the gripper
    # Plot the entire arm
    # More angles to check
    angles_check_link_0 = [np.pi/4, 0.0, 0.0, [0.0, 0.0, 0.0]]
    angles_check_link_0_1 = [np.pi/4, -np.pi/4, 0.0, [0.0, 0.0, 0.0]]
    angles_gripper_check = [np.pi/6.0, -np.pi/4, 1.5 * np.pi/4, [np.pi/3.0, -np.pi/8.0, np.pi/6.0]]

    # Actually set the matrices
    set_angles_of_arm_geometry(arm_geometry, angles_gripper_check)
    matrices = get_matrices_all_links(arm_geometry)

    # Print out the matrices (if you want)
    np.set_printoptions(precision=2, suppress=True)
    for i, m in enumerate(matrices):
        print(f"Matrix {i} is\n{m}")

    # Now actually plot - when you do the gripper grasp location (step 4) it will show up here
    fig2, axs2 = plt.subplots(1, 1, figsize=(8, 8))
    plot_complete_arm(axs2, arm_geometry, matrices)
    print("Done")
