#!/usr/bin/env python3

# The usual imports
import numpy as np
import matplotlib.pyplot as plt

# If this doesn't work, right click on top level folder and pick "mark folder" as source
import arm_forward_kinematics as afk


# --------------------------- Inverse Kinematics ------------------
# The goal of this part of the assignment is to move the grasp point to a target position.
#
# Option 1: Bare-bones gradient descent
# Option 2: Jacobians all the way down (with pseudo least squares to solve)
#
# Note: The point of this assignment is to get an understanding of how these techniques work. After this, you should
#  never (well, almost never) write your own gradient descent/IK solver
# Slides: https://docs.google.com/presentation/d/11gInwdUCRLz5pAwkYoHR4nzn5McAqfdktITMUe32-pM/edit?usp=sharing
#


# -------------------- Distance calculation --------------------
# Whether you're doing gradient descent or IK, you need to know the vector and/or distance from the current
#  grasp position to the target one. I've split this into two functions, one that returns an actual vector, the
#  second of which calls the first then calculates the distance. This is to avoid duplicating code.
def vector_to_goal(arm_with_angles, target):
    """
    Calculate the grasp pt for the current arm configuration. Return a vector going from the grasp pt to the target
    @param arm_with_angles - The arm geometry, as constructed in arm_forward_kinematics
    @param target - a 2x1 numpy array (x,y) that is the desired target point
    @return - a 2x1 numpy array that is the vector (vx, vy)
    """
    # TODO:
    #   Get the gripper/grasp location using get_gripper_location
    #   Calculate and return the vector
# YOUR CODE HERE


def distance_to_goal(arm_with_angles, target):
    """
    Length of vector - this function is zero when the gripper is at the target location, and gets bigger
    as the gripper moves away
    Note that, for optimization purposes, the distance squared works just as well, and will save a square root - but
    for the checks I'm using distance.
    @param arm_with_angles - The arm geometry, as constructed in arm_forward_kinematics
    @param target - a 2x1 numpy array (x,y) that is the desired target point
    @return: The distance between the gripper loc and the target
    """

    # TODO: Call the function above, then return the vector's length
# YOUR CODE HERE


def calculate_gradient(arm, angles, target):
    """
    Calculate the gradient (derivative of distance_to_goal) with respect to each link angle (and the wrist angle)
    @param arm - The arm geometry, as constructed in arm_forward_kinematics
    @param angles - A list of angles for each link, followed by a triplet for the wrist and fingers
    @param target - a 2x1 numpy array (x,y) that is the desired target point
    @return: f(x+h) - f(x) / h for all link angles (and the wrist angle) as a numpy array
    """
    # Use this value for h - it's small enough to be close to the correct derivative, but not so small that we'll
    #  run into numerical errors
    h = 1e-6

    # Derivatives - append each derivative to this list
    derivs = []

    # TODO
    # Step 1: First, calculate f(x) (the current distance)
    #   Don't forget to set the angles of the arm (afk.set_angles_of_arm_geometry)
    # Step 2: For each link angle (do the gripper last)
    #   Add h to the angle
    #   Calculate the new distance with the new angles
    #   Subtract h from the angle
    #   Calculate (f(x+h) - f(x)) / h and append that to the derivs list
    # Step 3: Do the wrist/gripper angle the same way (but remember, that angle
    #   is stored in angles[-1][0])
# YOUR CODE HERE
    return derivs


# ------------------------ Gradient descent -----------------
# Calculate the gradient
# Step size is big
# while step size still big
#    While grasp point not at goal/moving closer to goal
#      Try a step by gradient * step_size
#    Shrink step size
def gradient_descent(arm, angles, target, b_one_step=True) -> tuple:
    """
    Do gradient descent to move grasp point towards target
    @param arm - The arm geometry, as constructed in arm_forward_kinematics
    @param angles - A list of angles for each link, followed by a triplet for the wrist and fingers
    @param target - a 2x1 numpy array (x,y) that is the desired target point
    @param b_one_step - if True, return angles after one successful movement towards goal
    @ return if we got better and angles that put the grasp point as close as possible to the target, and number
            of iterations it took
    """

    # Outer loop - keep going until no progress made
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
    best_distance = distance_to_goal(arm, target)
    count_iterations = 0
    while b_keep_going and count_iterations < 1000:
        # First, calculate the gradiant with the current angles
        # TODO: Calculate the gradient with angles (don't for get to set the angles first)
# YOUR CODE HERE

        # This is the while loop where you keep "shrinking" the step size until you get closer to the goal (if
        #  you ever do)
        # Again, use a Boolean to tag if you actually got better
        b_took_one_step = False

        # Start with a step size of 1 - take one step along the gradient
        step_size = 1.0
        # Two stopping criteria - either never got better OR one of the steps worked
        while step_size > 0.005 and not b_took_one_step:
            # TODO:
            #  Try taking a step along the gradient
            #    For each angle
            #      new_angle = angle - step_size * gradient_for_that_angle
            #    Remember that the last element of the list is a list with 3 elements in it - wrist and the two fingers
            #  Calculate what the new distance would be with those angles
            #  We go in the OPPOSITE direction of the gradient because we want to DECREASE distance
            new_angles = []
# YOUR CODE HERE

            # Now we see how we did
            afk.set_angles_of_arm_geometry(arm, new_angles)
            new_dist = distance_to_goal(arm, target)

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
    gripper_loc = afk.get_gripper_location(arm_geometry)
    exp_gripper_loc = (-0.3536, 1.4098)
    if not np.isclose(gripper_loc[0], exp_gripper_loc[0], atol=0.01) or not np.isclose(gripper_loc[1], exp_gripper_loc[1], atol=0.01):
        print(f"Gripper loc, expected {exp_gripper_loc}, got {gripper_loc}")
    else:
        print(f"Success: got same gripper location")

    # The location to reach for
    target = (0.5, 1.5)

    # ------------------ Syntax and return value checks ----------------------
    # First check - vector to goal - see pic above, should point from green + to red one
    vec = vector_to_goal(arm_geometry, target)
    if not vec.shape[0] == 2:
        print(f"Expected a 2x1 vector, got {vec.shape}")
    elif not np.isclose(vec[0], 0.8535, atol=0.01) or not np.isclose(vec[1], 0.09019, atol=0.01):
        print(f"Expected (0.853, 0.090), got {vec}")
    else:
        print("Passed vec check")

    # Second check - distance to goal, should be length of vector
    dist = distance_to_goal(arm_geometry, target)
    if not np.isclose(dist, 0.858, atol=0.01):
        print(f"Expected 0.858, got {dist}")
    else:
        print("Passed distance check")

    # Third check - gradient (how distance changes as angles change)
    # Note: This assumes you used the h that was given
    grad = calculate_gradient(arm_geometry, angles_check, target)
    exp_grad = [0.9419, 0.4447, 0.2114, 0.0559]
    if not len(grad) == 4:
        print(f"Expected a 4x1 list, got {grad}")
    elif not np.all(np.isclose(grad, exp_grad, atol=0.01)):
        print(f"Expected {exp_grad} got {grad}")
    else:
        print("Passed gradient check")

    # Main check - do we get out the new angles? Note, this assumes that you haven't changed up the step size
    b_succ, angles_new, count = gradient_descent(arm_geometry, angles_check, target, b_one_step=True)
    ang_exp = [-0.942, 0.341, -0.211, [-0.841, 0, 0]]
    if not len(angles_new) == 4:
        print(f"Expected {ang_exp}, got {angles_new}")
    for a1, a2 in zip(angles_new, ang_exp):
        if not np.all(np.isclose(a1, a2, atol=0.01)):
            print(f"Expected {a1} got {a2}")
    if not b_succ:
        print(f"Expected successful/improvement, got none")

    # This will actually change the arm geometry based on the new angles
    afk.set_angles_of_arm_geometry(arm_geometry, angles_check)
    last_dist = distance_to_goal(arm_geometry, target)

    # For first round of for loop
    angles_new = angles_check
    print(f"Starting distance {last_dist}")
    for i in range(0, 3):
        b_succ, angles_new, count = gradient_descent(arm_geometry, angles_new, target, b_one_step=True)
        afk.set_angles_of_arm_geometry(arm_geometry, angles_new)
        dist = distance_to_goal(arm_geometry, target)
        if not b_succ:
            print(f"Expected successful/improvement after each iteration, got none")
        elif not dist < last_dist:
            print(f"Expected improvement, got none, dist {dist}")
        else:
            print(f"Passed iteration check {i}, dist {dist}")
        last_dist = dist

    if not np.isclose(dist, 0.1, atol=0.01):
        print(f"Expected distance to be close to 0.1, got {dist}, count {count}")

    # ----------------- full solve check ----------------------------
    # Check the run until no improvement version
    b_succ, angles_new, count = gradient_descent(arm_geometry, angles_check, target, b_one_step=False)

    # Set the new angles
    afk.set_angles_of_arm_geometry(arm_geometry, angles_new)
    dist = distance_to_goal(arm_geometry, target)
    if not b_succ:
        print(f"Expected successful/improvement, got none")
    elif not np.isclose(dist, 0.063, atol=0.01):
        print(f"Expected distance to be close to 0.063, got {dist}, count {count}")
    else:
        print(f"Passed full descent {dist:0.2}")

    # ----------------- Plot ----------------------------
    # Plot so you can see what it should look like
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    for i, a in enumerate([angles_check, angles_new]):
        # Set the angles
        afk.set_angles_of_arm_geometry(arm_geometry, a)
        # Actually get the matrices
        matrices = afk.get_matrices_all_links(arm_geometry)
        afk.plot_complete_arm(axs[i], arm_geometry, matrices)

        # Plot target too
        axs[i].plot(target[0], target[1], '+r')

    axs[0].set_title("Before gradient descent")
    axs[1].set_title(f"After gradient descent count {count} dist {dist:0.3}")

    print("Done")
