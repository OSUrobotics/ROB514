#!/usr/bin/env python3

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

# Slides for this assignment: https://docs.google.com/presentation/d/1iTi45y5AghMZRgStPX4mPdR7uYFQdRBjaekOW7ESTxM/edit?usp=sharing

import numpy as np
import matplotlib.pyplot as plt


# -------------------- Matrices ---------------------

# These first few functions are just convenience functions for making specific transformation matrices (translation,
#  rotation, scale). Also a way to refresh your memory with the mechanics of building a specific matrix... Normally,
#  you would never write these (numpy has versions of them - in particular, use scipy's rotation/quaternion class to handle
#  rotations in 3D)
#
#  The rest of this code is specifying what kind of matrix + parameters, getting parameters out of matrices
#
# A reminder that all matrices are 3x3 so that we can do translations (the upper left is the 2x2 matrix)

# -------------------- Creating specific types of matrices ---------------------
def make_scale_matrix(scale_x=1.0, scale_y=1.0):
    """Create a 3x3 scaling matrix
    @param scale_x - scale in x. Should NOT be 0.0
    @param scale_y - scale in y. Should NOT be 0.0
    @returns a 3x3 scaling matrix"""
    if np.isclose(scale_x, 0.0) or np.isclose(scale_y, 0.0):
        raise ValueError(f"Scale values should be non_zero {scale_x}, {scale_y}")

    mat = np.identity(3)
    # GUIDE: set the relevant values of mat
    # YOUR CODE HERE

    return mat


def make_translation_matrix(d_x=0.0, d_y=0.0):
    """Create a 3x3 translation matrix that moves by dx, dy
    @param d_x - translate in x
    @param d_y - translate in y
    @returns a 3x3 translation matrix"""

    mat = np.identity(3)
    # GUIDE: set the relevant values of mat
    # YOUR CODE HERE

    return mat


def make_rotation_matrix(theta=0.0):
    """Create a 3x3 rotation matrix that rotates counter clockwise by theta
    Note that it is convention to rotate counter clockwise - there's no mathematical reason for it
    @param theta - rotate by theta (theta in radians)
    @returns a 3x3 rotation matrix"""

    mat = np.identity(3)
    # GUIDE: set the relevant values of mat
    # YOUR CODE HERE

    return mat


# ---------------------------- Text versions of matrices ------------------------
#
# Rather than keep the actual matrix (which takes space, is difficult to understand, and also subject to numerical
#  error) it is more common in URDF files to specify matrics by their type and parameters.
#
# In our case, we're going to keep the text description of the matrix as a dictionary.
# Note: I've done all of these for you
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
    return {"type": "translate", "dx": dx, "dy": dy}


def make_matrix_from_sequence(seq):
    """ Turn an array of dictionaries with matrix type and params into a single 3x3 matrix
    Assumption: The first item in the list is the first transformation to apply to the object
    Remember: The matrix that is closest to the object vertices (the right most one) is the first applied matrix
       seq = (R, T)
        T R XYs   - R is applied first if matrices are multiplied in this order
    @param seq - list of transformations
    @return - a matrix"""

    mat = np.identity(3)
    for s in seq:
        # Doing this part for you - converting an entry in the sequence into a matrix
        if "translate" in s["type"]:
            next_mat = make_translation_matrix(s["dx"], s["dy"])
        elif "scale" in s["type"]:
            next_mat = make_scale_matrix(s["sx"], s["sy"])
        elif "rotate" in s["type"]:
            next_mat = make_rotation_matrix(s["theta"])
        else:
            raise ValueError(f"Expected one of translate, scale, rotate, got {s['type']}")
        # GUIDE: multiply next_mat by mat and store the result in mat
        #    (reminder: @ is matrix multiplication)
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

    # GUIDE:
    #  1) Create a vector for the x-axis and multiply it by the matrix. The LENGTH of the vector is the scale
    #     in x
    #  2) Repeat for the y-axis
    # Remember that when multiplying matrices by Vectors (as opposed to points) you should put 0 in that
    #   third coordinate, since vectors do not have a location
    # np.linalg.norm() will get the length of the vector
    # YOUR CODE HERE


def get_dx_dy_from_matrix(mat):
    """Where does the matrix translate 0,0 to?
    @param mat - the matrix
    @returns dx, dy - the transformed point 0,0"""

    # GUIDE:
    #  1) Multiply the point (0,0) by the matrix
    #  2) Return the point mat * pt
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

    # GUIDE:
    #  1) Set x_axis to be a unit vector pointing down the x axis
    #  2) Set y_axis to be a unit vector pointing down the y axis
    #  Multiply by the matrix to get the new "x" and "y" axes
    # YOUR CODE HERE


def get_theta_from_matrix(mat):
    """ Get the actual rotation angle from how the x-axis transforms
    @param mat - the matrix
    @return theta, the rotation amount in radians"""

    # GUIDE
    # Step 1) Use get_axes_from_matrix to get the x_axis,
    # Step 2) use arctan2 to turn the rotated x axis vector into an angle
    #   Use the x axis because theta for the x axis is 0 (makes the math easier)
    # Reminder: arctan2 takes (y, x)
    # YOUR CODE HERE


# -------------------------------- Check and test routines ------------------------------
def check_is_rotation(mat, b_print=False):
    """ Rotation matrices are orthonormal. Always a good idea to check if your rotation matrix is, indeed, a rotation
    matrix.
    Orthonormal: check(s) rows/columns are unit length and orthogonal <v, w> = 0 to each other
    Additional check: You can also use the fact that a rotation matrix's inverse is its transpose
    @param mat - the matrix
    @param b_print - True/False - if True, print out the reason the check failed
    @return True if within epsilon of rotation"""

    # Hint: Use get_axes_from_matrix to get the first and second rows from the 3x3 matrix
    # You might find numpy.linalg.norm and numpy.dot and numpy.isclose useful
    # GUIDE: Return TRUE if the matrix is orthonormal/rotation matrix
    #       Return FALSE otherwise
    #       If b_print_test is True, also print out why the rotation matrix failed
    # YOUR CODE HERE
    return True


# Check if flip/mirror
#   Normally, if all scale matrices are positive (almost never scale by a negative value) then the handed-ness of
#   the coordinate system is preserved (see eg https://en.wikipedia.org/wiki/Right-hand_rule). What this means for 2D
#   matrices is that the cross product of mat * x_axis and mat * y_axis is the same as the cross product of
#   the x_axis and the y_axis (i.e., [0, 0, 1]). If this is NOT the case, then the object ends up mirrored. Very
#   rarely is this intentional; it's usually the result of accidentally scaling by a negative value in one axis
def check_is_mirrored(mat):
    """ Check if the matrix mirrors/i.e. flips the axes around
    @param mat - the matrix
    @return True if the cross product of the transformed axes is [0, 0, -1]"""

    # GUIDE:
    #  Step 1: Get the transformed axes using the get_axes_from_matrix
    #  Step 2: Get the cross product of the two matrices (see np.cross). Also make sure you do x, y (order matters for cross product)
    #  Step 3: Check that the resulting vector points in the positive z direction (x and y values are 0, z is positive)
    #  Note: Only the DIRECTION matters - not how long the vector is
    # YOUR CODE HERE


# Check if skewed/not angle-preserving
#   Normally, robotics only uses combinations of matrices that preserve angles - i.e., the angles between two vectors
#  doesn't change. If this is not true, then your robot will look like it got sat on by an elephant.
#  There are some mathematical conditions for this (see, eg, https://en.wikipedia.org/wiki/Transformation_matrix),
#  but broadly speaking, if you do a scale, followed by any number of rotations/translations, you'll get an affine
#  (angle preserving) transformation
def check_preserves_angles(mat):
    """ Check if the matrix preserves angles
    @param mat - the matrix
    @return True if angles are preserved"""

    # GUIDE:
    #  Step 1: Get the transformed axes using the get_axes_from_matrix
    #  Step 2: Get the angle between the transformed axes (remember that cos(angle) = dot(u,v) / (||u|| ||v||)
    #    see https://www.wikihow.com/Find-the-Angle-Between-Two-Vectors
    #  Step 3: Check that the angle between them is 90 degrees (reminder, numpy does everything in radians)
    #    Actually, you can just check that the dot product is close to 0
    # YOUR CODE HERE


# Check/test functions for autograder
#  Nothing for you to write here, but you should understand what these routines are checking
def test_translation_matrix():
    """ Raises an error if the matrix/return value is incorrect"""
    # Easy check - is the translation matrix correct? Is the get dx, dy back out correct?
    dx_dys = ((0.2, -1), (-0.1, 0.3), (2, 0.2))
    for dx, dy in dx_dys:
        mat = make_translation_matrix(dx, dy)
        if not np.isclose(mat[0, 2], dx) or not np.isclose(mat[1, 2], dy):
            raise ValueError(f"Matrix not built correctly {mat}, ({dx}, {dy})")

        xy_trans = mat @ np.transpose(np.array([0.0, 0.0, 1.0]))
        if not np.isclose(xy_trans[0], dx) or not np.isclose(xy_trans[1], dy) or not np.isclose(xy_trans[2], 1.0):
            raise ValueError(f"Matrix not built correctly {mat}, ({dx}, {dy})")

        x_back, y_back = get_dx_dy_from_matrix(mat)
        if not np.isclose(x_back, dx) or not np.isclose(y_back, dy):
            raise ValueError(f"Matrix not built correctly {mat}, ({dx}, {dy})")


def test_scale_matrix():
    """ Raises an error if the matrix/return value is incorrect"""
    # Easy check - is the scaling matrix correct? Is the get sx, sy back out correct?
    sx_sys = ((1.2, 1.0), (0.1, 0.3), (2, 0.2))
    for sx, sy in sx_sys:
        mat = make_scale_matrix(sx, sy)
        if not np.isclose(mat[0, 0], sx) or not np.isclose(mat[1, 1], sy):
            raise ValueError(f"Matrix not built correctly {mat}, ({sx}, {sy})")

        xy_scale = mat @ np.transpose(np.array([1.0, 1.0, 0.0]))
        if not np.isclose(xy_scale[0], sx) or not np.isclose(xy_scale[1], sy) or not np.isclose(xy_scale[2], 0.0):
            raise ValueError(f"Matrix not built correctly {mat}, ({sx}, {sy})")

        sx_back, sy_back = get_sx_sy_from_matrix(mat)
        if not np.isclose(sx_back, sx) or not np.isclose(sy_back, sy):
            raise ValueError(f"Matrix extraction not correct {mat}, ({sx}, {sy}), ({sx_back}, {sy_back})")


def test_rotation_matrix():
    """ Raises an error if the matrix/return value is incorrect"""
    # Easy check - is the scaling matrix correct? Is the get sx, sy back out correct?
    thetas = (0.0, np.pi / 2.0, np.pi * 0.1)
    for theta in thetas:
        mat = make_rotation_matrix(theta)
        mat_inv = make_rotation_matrix(-theta)
        mat_ident = mat @ mat_inv
        if np.count_nonzero(np.isclose(mat_ident, np.identity(3))) != 9:
            raise ValueError("Rotation matrix inverse not transpose {theta}, {mat}")

        if not check_is_rotation(mat):
            raise ValueError("Rotation matrix not orthonormal {theta}, {mat}")

        theta_back = get_theta_from_matrix(mat)
        if not np.isclose(theta, theta_back):
            raise ValueError(f"Matrix not built correctly {mat}, {theta}, {theta_back}")


def test_matrices():
    """ Raises an error if the matrix/return value is incorrect"""
    test_scale_matrix()
    test_translation_matrix()
    test_rotation_matrix()
    return True


def test_checks(b_print=False):
    """ Test the check routines"""
    mat_not_orthonormal = np.identity(3)
    mat_not_orthonormal[0, 0] = 2.0
    if check_is_rotation(mat_not_orthonormal, b_print):
        raise ValueError(f"Matrix {mat_not_orthonormal} is not orthonormal, should fail length check")

    mat_not_orthonormal[0, 0] = 1.0
    mat_not_orthonormal[1, 1] = np.sqrt(1.0 - 0.1 * 0.1)
    mat_not_orthonormal[1, 0] = 0.1

    if check_is_rotation(mat_not_orthonormal, b_print):
        raise ValueError(f"Matrix {mat_not_orthonormal} is not orthonormal, should fail orthogonal check")

    mat_mirrored = make_scale_matrix(2.0, -1.2)
    if not check_is_mirrored(mat_mirrored):
        raise ValueError(f"Matrix {mat_mirrored} is mirrored, should return True")

    mat_skewed = make_scale_matrix(2.0, 1.2) @ make_rotation_matrix(0.25)
    if  check_preserves_angles(mat_skewed):
        raise ValueError(f"Matrix {mat_skewed} does NOT preserve angles, should return False")

    mat_is_ok = make_rotation_matrix(np.pi/3.0) @ make_translation_matrix(0.2, -0.3) @ make_rotation_matrix(-np.pi/2.0) @ make_scale_matrix(0.2, 2.0)
    if check_is_mirrored(mat_is_ok):
        raise ValueError(f"Matrix {mat_is_ok} is NOT mirrored, should return False")
    if not check_preserves_angles(mat_is_ok):
        raise ValueError(f"Matrix {mat_is_ok} is angle-preserving, should return True")
    if check_is_rotation(mat_is_ok):
        raise ValueError(f"Matrix {mat_is_ok} is NOT orthonormal, should return False")

    mat_is_ortho = make_rotation_matrix(np.pi/3.0) @ make_rotation_matrix(-np.pi/2.0)
    mat_is_not_ortho = make_rotation_matrix(np.pi/3.0) @ make_translation_matrix(0.2, -0.3) @ make_rotation_matrix(-np.pi/2.0)
    if not check_is_rotation(mat_is_ortho):
        raise ValueError(f"Matrix {mat_is_ortho} is orthonormal, should return True")
    if check_is_rotation(mat_is_not_ortho):
        raise ValueError(f"Matrix {mat_is_not_ortho} is NOT orthonormal, should return False")

    return True


# ------------------------------- Plot code -------------------------------------
def make_pts_representing_circle(n_pts=25):
    """Create a 3xn_pts matrix of the points on a circle
    @param n_pts - number of points to use
    @return a 3xn numpy matrix"""

    ts = np.linspace(0, np.pi * 2, n_pts)
    # GUIDE: make a 3 x n_pts array of points for the circle
    #   These are the x,y points of a unit circle centered at the origin
    #   These are the points that we will draw, both in their original location and in their transformed location
    # Step 1: Make a 3 x n_pts numpy array - I like to use np.ones, because it sets the homogenous coordinate for me
    # Step 2: Set the x values of the array to be cos(t) for the ts given above (you don't need a for loop for this,
    #   see numpy array math
    # Step 3: Do the same for the y values, but set to sin(t)
    # YOUR CODE HERE
    return pts


def plot_axes_and_box(axs, mat):
    """Plot the original coordinate system (0,0 and x,y axes) and transformed coordinate system
    @param axs - figure axes
    @param mat - the matrix"""

    # Initial location - center of coordinate system
    axs.plot(0, 0, '+k')
    axs.plot([0, 1], [0, 0], '-r')
    axs.plot([0, 0], [0, 1], '-b')

    # Draw a box around the world to make sure the plots stay the same size
    axs.plot([-5, 5, 5, -5, -5], [-5, -5, 5, 5, -5], '-k')

    # Moved coordinate system - draw the moved coordinate system and axes
    dx, dy = get_dx_dy_from_matrix(mat)
    x_axis, y_axis = get_axes_from_matrix(mat)

    axs.plot(dx, dy, 'Xb', markersize=5)
    axs.plot([dx, dx + x_axis[0]], [dy, dy + x_axis[1]], '-r', linewidth=2)
    axs.plot([dx, dx + y_axis[0]], [dy, dy + y_axis[1]], '-b', linewidth=2)

    # This makes sure the x and y axes are scaled the same
    axs.axis('equal')


def plot_axes_and_circle(axs, mat):
    """Plot the original coordinate system (0,0 and x,y axes) and transformed coordinate system and transformed circle
    @param axs - figure axes
    @param mat - the matrix"""

    # The axes and the box around the world
    plot_axes_and_box(axs, mat)

    # Make a circle
    pts = make_pts_representing_circle(25)

    # Draw circle
    axs.plot(pts[0, :], pts[1, :], ':g')

    # GUIDE: Transform circle by mat and put new points in pts_moved
    # YOUR CODE HERE
    axs.plot(pts_moved[0, :], pts_moved[1, :], ':g')


def plot_zigzag(axs, mat):
    """Plot a zigzag before and after the transform
    @param axs - figure axes
    @param mat - the matrix"""

    # zigzag geometry
    pts = np.ones((3, 5))
    for i in range(0, 5):
        if i // 2:
            pts[0, i] = -1
        if i % 2:
            pts[1, i] = -1

    # Draw zigzag
    axs.plot(pts[0, :], pts[1, :], linestyle='dashed', color='grey')

    # Draw moved zigzag
    pts_moved = mat @ pts
    axs.plot(pts_moved[0, :], pts_moved[1, :], linestyle='dashed', color='grey')
    return pts


def example_order_matters():
    """ Make the plot for rotation-translation versus translation-rotation"""
    # Make the plot that shows the difference between rotate-translate and translate-rotate
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    seq_rot_trans = [{"type":"rotate", "theta": np.pi/4.0},
                     {"type":"translate", "dx": 1, "dy": 2}]

    mat = make_matrix_from_sequence(seq_rot_trans)
    axs[0, 0].set_title("Rot trans")
    plot_axes_and_circle(axs[0, 0], mat)

    # Reverse the order of operations
    # GUIDE: Now create the matrix in the reverse order - try to predict what this will look like
    #   Set mat to be a translation, rotation matrix (same params as above)
    axs[0, 1].set_title("Trans rot")
    # YOUR CODE HERE
    plot_axes_and_circle(axs[0, 1], mat)

    # GUIDE Now do a matrix (mat) that is a scale 0.5,2.0, rotate pi/4, translate (1,2)
    # YOUR CODE HERE

    axs[1, 0].set_title("Scl rot trans")
    plot_axes_and_circle(axs[1, 0], mat)

    # Reverse the order of operations
    # GUIDE Now do a matrix (mat) that is the REVERSE of the scale, rotate, translate
    # YOUR CODE HERE
    axs[1, 1].set_title("Trans rot scl")

    plot_axes_and_circle(axs[1, 1], mat)


def example_weird_geometry():
    """ Create a mirrored and a non-angle preserving example """
    # Make the plot that shows the difference between rotate-translate and translate-rotate
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    # GUIDE: Make seq_mirrored so that the x,y axes are flipped, and the x axis is twice as big as it was before.
    #  Draw the the flipped geometry at 2.5 2.5
    #  (see mirrored figure in slides https://docs.google.com/presentation/d/1iTi45y5AghMZRgStPX4mPdR7uYFQdRBjaekOW7ESTxM/edit?usp=sharing)
    # Should be a scale followed by a translate
    # YOUR CODE HERE

    mat = make_matrix_from_sequence(seq_mirrored)
    # print(f"{mat}")
    axs[0].set_title("Mirrored")
    plot_axes_and_circle(axs[0], mat)
    plot_zigzag(axs[0], mat)

    # GUIDE: Make seq_skew so that the axes (red blue) are no longer 90 degrees. There are multiple solutions to this, btw.
    #  One of the simplest is to rotate then scale x differently than y
    #  Draw the the flipped geometry at 2.5 2.5 (see skewed figure in slides https://docs.google.com/presentation/d/1iTi45y5AghMZRgStPX4mPdR7uYFQdRBjaekOW7ESTxM/edit?usp=sharing)
    # YOUR CODE HERE

    mat = make_matrix_from_sequence(seq_skew)
    # print(f"{mat}")
    axs[1].set_title("Skewed")
    plot_axes_and_circle(axs[1], mat)
    plot_zigzag(axs[1], mat)


def example_uncentered_geometry():
    """ Same matrix tranforms - but the object in a different place """
    # First plot is the "normal" one
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    pts_circle = make_pts_representing_circle(25)
    pts_zigzag = plot_zigzag(axs[0], np.identity(3))

    # GUIDE: create the pts_circle_* and pts_zigzag_* by
    #  Moving the original geometry so that the origin is at the lower left corner of the circle
    #  Rotating the original geometry so that the x axis is "up"
    # Note: You can use the make_x_matrix commands to move the points
    # YOUR CODE HERE

    seq_scl_rot_trans = [{"type":"scale", "sx":0.5, "sy":0.75},
                         {"type":"rotate", "theta": np.pi/3.0},
                         {"type":"translate", "dx": -1, "dy": 2.0}]
    mat = make_matrix_from_sequence(seq_scl_rot_trans)

    # First plot - what happens if the circle and zig zag are centered
    plot_axes_and_circle(axs[0], mat)
    plot_zigzag(axs[0], mat)
    axs[0].set_title("Geometry centered")

    # Fancy Python looping - create 3 lists, one with the name to put in the title, the second and third
    #   with the three point matrices created above.
    list_names = ['Origin lower left', 'x up', 'x up and lower left origin']
    list_pts_circle = [pts_circle_lower_left_origin, pts_circle_x_up, pts_circle_x_up_lower_left_origin]
    list_pts_zigzag = [pts_zigzag_lower_left_origin, pts_zigzag_x_up, pts_zigzag_x_up_lower_left_origin]
    for i, (n, c, z) in enumerate(zip(list_names, list_pts_circle, list_pts_zigzag)):
        # Plot the points in their original location
        axs[i+1].plot(c[0, :], c[1, :], linestyle='solid', color='green')
        axs[i+1].plot(z[0, :], z[1, :], linestyle='dashed', color='grey')

        # Plot the points in their scale-rotate-translated location (mat)
        pts_moved = mat @ c
        axs[i+1].plot(pts_moved[0, :], pts_moved[1, :], ':g')
        pts_moved = mat @ z
        axs[i+1].plot(pts_moved[0, :], pts_moved[1, :], linestyle='dashed', color='grey')

        # Draw the axes and the box
        plot_axes_and_box(axs[i+1], mat)
        axs[i+1].set_title(n)


if __name__ == '__main__':
    # Call the test routines
    test_matrices()

    test_checks(b_print=True)

    # Order of matrices matters
    example_order_matters()

    # Mirrored and not angle preserving
    example_weird_geometry()

    # Object not where you expect it to be
    example_uncentered_geometry()

    # Depending on if your mac, windows, linux, and if interactive is true, you may need to call this to get the plt
    # windows to show
    plt.show()
    print("Done")

