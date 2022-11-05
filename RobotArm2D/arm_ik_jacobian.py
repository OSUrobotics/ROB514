#!/usr/bin/env python3

# The usual imports
import numpy as np
import matplotlib.pyplot as plt

# --------------------------- Inverse Kinematics ------------------
# The goal of this part of the assignment is to move the grasp point to a target position, using the Jacobian.
##
# Note: The point of this assignment is to get an understanding of how these techniques work. After this, you should
#  never (well, almost never) write your own gradient descent/Jacobian/IK solver
# Slides: https://docs.google.com/presentation/d/11gInwdUCRLz5pAwkYoHR4nzn5McAqfdktITMUe32-pM/edit?usp=sharing
#

import matrix_transforms as mt
import arm_forward_kinematics as afk
import arm_ik_gradient as ik_gradient


# ----------------------------- Practice with calculating Jacobian for one link -------------------------------------------

def practice_jacobian():
    """ An example problem of an arm with radius 3 currently at angle theta = 0.2
    This is just to check that you can calculate the Jacobian for one link before doing the matrix version
     If you have trouble with this, get the TA to give you the solution"""
    radius = 3
    theta = 0.2

    pt_end = [radius * np.cos(theta), radius * np.sin(theta)]

    # TODO: Create a 3D vector to the end point (r cos(theta), r sin(theta), 0)
    #   Needs to be 3D for cross product to work
# YOUR CODE HERE

    # The z vector we spin around
    omega_hat = [0, 0, 1]

    # TODO: Take the cross product of omega_hat and r
    #  The result should always be 0 in 3rd component for a 2D vector
    #  Order matters for cross products...
# YOUR CODE HERE


    # TODO: Build the Jacobian, which in this case is a 2x1 matrix
    # This matrix takes changes in the angles to changes in x,y
    #   ... and is just the first two components of omega hat cross r
    # TODO: Set the column of the matrix to be omega hat cross r
    jacobian_matrix = np.zeros([2, 1])
# YOUR CODE HERE

    # Now we'll set up a linear equation solve that looks like
    #  A x = b  (np.linalg.lstsq)
    # Where A is the Jacobian we just built, b is the desired change in x,y,
    #   and x is the angle we're solving form

    # TODO: Create a 2x1 matrix that stores the desired change in x and change in y
    #  ... in this case, use -0.01 and -0.1
    # Desired x,y change
    b_matrix = np.zeros([2, 1])
# YOUR CODE HERE
    pt_new_end = [pt_end[0] - 0.01, pt_end[1] - 0.1]

    # TODO: Solve the matrix using np.linalg.lstsq. Should return a 1x1 matrix with an angle change
    #   Note: Use rcond=None
    d_ang = np.zeros([1, 1])
# YOUR CODE HERE

    # Check result of solve - should be the same as dx_dy
    res = jacobian_matrix @ d_ang[0]

    # The actual point you end up at if you change the angle by that much
    pt_moved = [radius * np.cos(theta + d_ang[0][0]), radius * np.sin(theta + d_ang[0][0])]

    print(f"Delta angle {d_ang[0]}, should be -0.32")
    print(f"New point and moved point should be the close to the same")
    print(f"Old point: {pt_end}\nNew pt: {pt_new_end}\nMoved pt: {pt_moved}")

    return d_ang


# ----------------------------- Jacobian -------------------------------------------
#  Break the IK solve into two pieces, calculating the Jacobian matrix and then
#   Solving it with specific desired changes in x and y -> give desired changes
#     in angles that will (hopefully) get those changes in location

def calculate_jacobian_numerically(arm, angles):
    """ Use the (f(x+h) - f(x))/h approximation to calculate the Jacobian
    @param arm - The arm geometry, as constructed in arm_forward_kinematics
    @param angles - A list of angles for each link, followed by a triplet for the wrist and fingers
    @return 2xn Jacobian matrix that maps changes in angles to changes in x,y """
    # One column for each link plus the wrist, two rows, one for x, one for y
    jacob = np.zeros([2, len(arm) - 1])

    # Use this value for h - it's small enough to be close to the correct derivative, but not so small that we'll
    #  run into numerical errors
    h = 1e-6

    # TODO
    # Step 1: First, calculate gripper loc() (the current gripper location)
    # Step 2: For each link angle (do the gripper last)
    #   Add h to the angle
    #   Calculate the new location with the new angles
    #   Subtract h from the angle
    #   Calculate (f(x+h) - f(x)) / h for the x and y location, and put it in the ith column
    # Step 3: Do the wrist/gripper angle the same way (but remember, that angle
    #   is stored in angles[-1][0])
# YOUR CODE HERE
    return jacob

def calculate_jacobian(arm, angles):
    """
    Calculate the Jacobian from the given angles and the lengths in the arm
    Start with the wrist and work *backwards*, calculating Ri @ Ti @ previous_matrix
    The vector r from the practice Jacobian problem above is just the last column of that matrix
    This nethod is OPTIONAL, but should return the same thing as calculate_jacobian_numerically
    @param arm - The arm geometry, as constructed in arm_forward_kinematics
    @param angles - A list of angles for each link, followed by a triplet for the wrist and fingers
    @return 2xn Jacobian matrix that maps changes in angles to changes in x,y """

    # One column for each link plus the wrist (ignores base)
    jacob = np.zeros([2, len(arm) - 1])

    # TODO: To make things simpler, first build two lists that have the link lengths and angles in REVERSE order
    # Note that python has a handy reverse method for lists
    #  This is Python-ese for getting out the lengths from the list of dictionaries
    #  Don't forget the wrist
    lengths_links = [link["Arm length"] for link in arm[1:-1]]
    lengths_links.append(arm[-1][0]["Grasp"])

    # TODO: reverse the length list, make the reversed angles list from angles
# YOUR CODE HERE

    # We rotated the base so the arm points up - so the last link angle needs to have that rotation added
    angles_links[-1] += np.pi / 2.0

    # TODO: Now work backwards, calculating R @ T @ mat_accum
    #   In each iteration, update mat_accum THEN multiply by the rotation up
    #    to that joing and THEN extract the vector r then do omega_hat cross r and put the result in the
    #    column of the matrix
    mat_accum = np.identity(3)
    # The z vector we spin around
    omega_hat = [0, 0, 1]

    # More python-ese - this gets each of the angles/lengths AND an enumeration variable i
    total_angles = np.sum(np.array(angles_links))
    for i, (ang, length) in enumerate(zip(angles_links, lengths_links)):
        # TODO:
        #   mat_accum is updated by rot(angle) @ trans(length, 0) @ mat_accum
        #     This gets the end point for all the links, working backwards from the wrist
        #   To get r, you need to rotate this by the sum of all the angles UP TO this angle
        #      mat_r = rot(sum all link angles up to this angle) @ mat_accum
        #     Get r from mat_r (the last column)
        #       Do omega_hat cross r
        #    Put the result in the n-i column in jacob - i.e., wrist should go in the last column in jacob
# YOUR CODE HERE
    return jacob


def solve_jacobian(jacobian, vx_vy):
    """ Do the pseudo inverse of the jacobian
    @param - jacobian - the 2xn jacobian you calculated from the current joint angles/lengths
    @param - vx_vy - a 2x1 numpy array with the distance to the target point (vector_to_goal)
    @return - changes to the n joint angles, as a 1xn numpy array"""

    # TODO: Call numpy's linear algebra least squares (linalg.lstsq) routine to calculate A x = b
    # Reminder: lstsq returns a tuple. See docs. The returned matrix is in the first part of the tuple
# YOUR CODE HERE
    return delta_angles


def jacobian_follow_path(arm, angles, target, b_one_step=True):
    """
    Use jacobian to calculate angles that move the grasp point towards target. Instead of taking 'big' steps we're
    going to take small steps along the vector (because the Jacobian is only valid around the joint angles)
    @param arm - The arm geometry, as constructed in arm_forward_kinematics
    @param angles - A list of angles for each link, followed by a triplet for the wrist and fingers
    @param target - a 2x1 numpy array (x,y) that is the desired target point
    @param b_one_step - if True, return angles after one successful movement towards goal
    @ return if we got better and angles that put the grasp point as close as possible to the target, and number
            of iterations it took
    """

    # Outer loop - move towards the target goal in steps of d_step
    #   Or - if b_one_step is True - stop after one loop

    # I like to use a boolean variable for this type of while loop, so you can track all of the places
    #  where you want to bail/keep going
    # Just set it to be "False" when you reach an end condition
    b_keep_going = True

    # This is a variable that you will set when you actually got better/closer. This just makes it easier
    #  to return/debug
    b_found_better = False

    # Assumption: angles is always the current, best angles, and best_distance is the distance for those angles
    # Stopping conditions - I always like to add a count variable and stop after n loops - safety measure
    afk.set_angles_of_arm_geometry(arm, angles)
    best_distance = ik_gradient.distance_to_goal(arm, target)
    count_iterations = 0

    d_step = 0.05
    while b_keep_going and count_iterations < 1000:

        # This is the vector to the target. Take maximum 0.05 of a step towards the target
        vec_to_target = ik_gradient.vector_to_goal(arm, target)
        vec_length = np.linalg.norm(vec_to_target)
        if vec_length > d_step:
            # Shorten the step
            vec_to_target *= d_step / vec_length
        elif np.isclose(vec_length, 0.0):
            # we can stop now...
            b_keep_going = False

        # TODO: Use the Jacobian to calculate angle changes that will move the end effector along vec_to_target
        #  Do in two parts - calculate the jacobian (calculate_jacobian)
        #   then solve it (solve_jacobian) with the vector (ik_gradient.vector_to_goal)
        #  You can use calculate_jacobian OR calculate_jacobian_numerically

        delta_angles = np.zeros(len(angles))
# YOUR CODE HERE

        # This rarely happens - but if the matrix is degenerate (the arm is in a straight line) then the angles
        #  returned from solve_jacobian will be really, really big. The while loop below will "fix" this, but this
        #  just shortcuts the whole problem. There are far, far better ways to deal with this
        avg_ang_change = np.linalg.norm(delta_angles)
        if avg_ang_change > 100:
            delta_angles *= 0.1 / avg_ang_change
        elif avg_ang_change < 0.1:
            delta_angles *= 0.1 / avg_ang_change

        # Similar to gradient descent, don't move if this doesn't help
        # This is the while loop where you keep "shrinking" the step size until you get closer to the goal (if
        #  you ever do)
        # Again, use a Boolean to tag if you actually got better
        b_took_one_step = False

        # Start with a step size of 1 - take one step along the gradient
        step_size = 1.0
        # Two stopping criteria - either never got better OR one of the steps worked
        while step_size > 0.005 and not b_took_one_step:

            # TODO:
            #  Try taking a step in the direction of the jacobian
            #    For each angle
            #      new_angle = angle + step_size * delta_angles
            #  Calculate what the new distance would be with those angles
            new_angles = []
# YOUR CODE HERE
            # Get the new distance with the new angles
            afk.set_angles_of_arm_geometry(arm, new_angles)
            new_dist = ik_gradient.distance_to_goal(arm, target)

            # TODO:
            #   If the new distance is larger than the best distance, decrease the step size (I suggest cutting it in half)
            #   Otherwise, set b_took_one_step to True (this will break out of the loop) and
            #     set angles to be new_angles and best_distance to be new_distance
            #     set b_found_better to be True
# YOUR CODE HERE
            # Count iterations
            count_iterations += 1

        # We can stop if we're close to the goal
        if np.isclose(best_distance, 0.0):
            b_keep_going = False

        # End conditions - b_one_step is true  - don't do another round
        #   OR we didn't take a step (b_took_one_step)
        if b_one_step or not b_took_one_step:
            b_keep_going = False

    # Return the new angles, and whether or not we ever found a better set of angles
    return b_found_better, angles, count_iterations


if __name__ == '__main__':
    np.printoptions(precision=4)

    #  Do this first to make sure you know how to calculate a Jacobian for 1 link
    print("Checking practice jacobian")
    ang = practice_jacobian()
    exp_ang = -0.032
    if not np.isclose(ang, exp_ang, atol=0.01):
        print(f"Practice jacobian: Expected {exp_ang}, got {ang}")
    else:
        print(f"Passed practice jacobian test")

    # ----------------------------- Setup ------------------------
    # Create the arm geometry
    base_size_param = (1.0, 0.5)
    link_sizes_param = [(0.5, 0.25), (0.3, 0.1), (0.2, 0.05)]
    palm_width_param = 0.1
    finger_size_param = (0.075, 0.025)

    arm_geometry = afk.create_arm_geometry(base_size_param, link_sizes_param, palm_width_param, finger_size_param)

    # Set some initial angles
    angles_check = [0.0, np.pi/4, 0, [-np.pi/4.0, np.pi/4.0, -np.pi/4.0]]
    afk.set_angles_of_arm_geometry(arm_geometry, angles_check)

    # The location to reach for
    target = (0.5, 1.5)

    # ------------------ Syntax and return value checks ----------------------

    # First check - calculating the jacobian (2 x 4 matrix) calculated numerically
    jacob = calculate_jacobian_numerically(arm_geometry, angles_check)
    exp_jacob = np.array([[-0.9098, -0.4098, -0.1977, -0.0562], [-0.3535, -0.3535, -0.1414, 0.0]])
    if not np.all(np.isclose(jacob, exp_jacob, atol=0.01)):
        print(f"Expected jacob\n{exp_jacob}, got\n{jacob}\n{np.isclose(jacob, exp_jacob, atol=0.01)}")
    else:
        print("Passed numerical jacob test")

    # Second check (optional) - calculating the jacobian using matrices (2 x 4 matrix)
    jacob = calculate_jacobian(arm_geometry, angles_check)
    if not np.all(np.isclose(jacob, exp_jacob, atol=0.01)):
        print(f"Expected jacob\n{exp_jacob}, got\n{jacob}\n{np.isclose(jacob, exp_jacob, atol=0.01)}")
    else:
        print("Passed matrix jacob test")

    # Third check - pseudo inverse 1 x 5 matrix of angle changes needed to get the desired x,y change
    delta_angles = solve_jacobian(np.array(exp_jacob), np.array([0.5, -0.2]))
    exp_delta_angles = np.array([-1.456, 1.819,  0.506, -0.368])
    if not np.all(np.isclose(delta_angles, exp_delta_angles, atol=0.01)):
        print(f"Expected delta angles\n{exp_delta_angles}, got\n{delta_angles}")
    else:
        print("Passed solve jacobian test")

    # ----------------- Do one step check ----------------------------
    # Main check - do we get out the new angles? Note, this assumes that you haven't changed up the step size
    b_succ, angles_new, count = jacobian_follow_path(arm_geometry, angles_check, target, b_one_step=True)
    ang_exp = [-0.0864, 0.8506, 0.01585, [-0.8024, 0, 0]]
    if not len(angles_new) == 4:
        print(f"Expected {ang_exp}, got {angles_new}")
    for a1, a2 in zip(angles_new, ang_exp):
        if not np.all(np.isclose(a1, a2, atol=0.01)):
            print(f"Expected angle {a2} got {a1}")
    if not b_succ:
        print(f"Expected successful/improvement, got none")

    # ----------------- full solve check ----------------------------
    # Check the run until no improvement version
    b_succ, angles_new, count = jacobian_follow_path(arm_geometry, angles_check, target, b_one_step=False)
    afk.set_angles_of_arm_geometry(arm_geometry, angles_new)
    dist = ik_gradient.distance_to_goal(arm_geometry, target)
    if not b_succ:
        print(f"Expected successful/improvement, got none")
    elif not np.isclose(dist, 0.062, atol=0.01):
        print(f"Expected distance to be close to 0.063, got {dist}, count {count}")
    else:
        print(f"Passed full solve check, dist {dist:0.2}")

    # ----------------- Plot ----------------------------
    # Plot so you can see what it should look like
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    for i, a in enumerate([angles_check, angles_new]):
        # Actually get the matrices
        afk.set_angles_of_arm_geometry(arm_geometry, a)
        matrices = afk.get_matrices_all_links(arm_geometry)
        afk.plot_complete_arm(axs[i], arm_geometry, matrices)

        # Plot target too
        axs[i].plot(target[0], target[1], '+r')

    axs[0].set_title("Before Jacobian descent")
    axs[1].set_title(f"After Jacobian descent count {count} dist {dist:0.2}")

    print("Done")


