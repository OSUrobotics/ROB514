#!/usr/bin/env python3

# The usual imports
import numpy as np

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

import matrix_transforms as mt
from RobotArm2D import arm_forward_kinematics as afk


# -------------------- Distance calculation --------------------
# Whether you're doing gradient descent or IK, you need to know the vector and/or distance from the current
#  grasp position to the target one. I've split this into two functions, one that returns an actual vector, the
#  second of which calls the first then calculates the distance. This is to avoid duplicating code.
def vector_to_goal(angles, arm, target):
    """
    Move the arm to have the given angles, then calculate the grasp pt. Return a vector going from the grasp pt to the target
    Note that it is bad Pythonese form to change the angles in the arm...
    @param angles - A list of angles for each link, followed by a triplet for the wrist and fingers
    @param arm - The arm geometry, as constructed in arm_forward_kinematics
    @param target - a 2x1 numpy array (x,y) that is the desired target point
    @return - a 2x1 numpy array that is the vector (vx, vy)
    """
    # TODO:
    #   Call set_angles_of_arm_geometry to set the angles
    #   Get the gripper/grasp location using get_gripper_location
    #   Calculate and return the vector
# YOUR CODE HERE


def distance_to_goal(angles, arm, target):
    """
    Length of vector - this function is zero when the gripper is at the target location, and gets bigger
    as the gripper moves away
    Note that, for optimization purposes, the distance squared works just as well, and will save a square root.
    @param angles - A list of angles for each link, followed by a triplet for the wrist and fingers
    @param arm - The arm geometry, as constructed in arm_forward_kinematics
    @param target - a 2x1 numpy array (x,y) that is the desired target point
    @return: The distance
    """

    # TODO: Call the function above, then return the vector's length
# YOUR CODE HERE


def calculate_gradient(angles, arm, target):
    """
    Calculate the gradient (derivative of distance_to_goal) with respect to each link angle (and the wrist angle)
    @param angles - A list of angles for each link, followed by a triplet for the wrist and fingers
    @param arm - The arm geometry, as constructed in arm_forward_kinematics
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
# Step size is big
# while step size still big
#    While grasp point not at goal (and we're still moving towards goal)
#      Calculate the gradient
#      Try a step by gradient * step_size
#    Shrink step size
def gradient_descent(angles, arm, target, b_one_step=True):
    """
    Do gradient descent to move grasp point towards target
    @param angles - A list of angles for each link, followed by a triplet for the wrist and fingers
    @param arm - The arm geometry, as constructed in arm_forward_kinematics
    @param target - a 2x1 numpy array (x,y) that is the desired target point
    @param b_one_step - if True, return angles after one successful movement towards goal
    @ return angles that put the grasp point as close as possible to the target
    """
    step_size = 1.0
    best_distance = distance_to_goal(angles, arm, target)
    while step_size > 0.05:
        # TODO: Calculate the gradient
# YOUR CODE HERE

        # TODO:
        #  Try taking a step along the gradient
        #    For each angle
        #      new_angle = angle - step_size * gradient_for_that_angle
        #  Calculate what the new distance would be with those angles
        #  We go in the OPPOSITE direction of the gradient because we want to DECREASE distance
        new_angles = []
# YOUR CODE HERE
        new_dist = distance_to_goal(new_angles, arm, target)

        # TODO:
        #   If the new distance is larger than the best distance, decrease the step size
        #   Otherwise, if b_one_stop is True, return the new angles
        #          if b_one_step is False, set angles to be new_angles and best_distance to be new_distance
# YOUR CODE HERE

    # We never got better - return the original angles
    return angles


# ----------------------------- Jacobians -------------------------------------------

def practice_jacobian():
    """ An example problem of an arm with radius 3 currently at angle theta = 0.2
    This is just to check that you can calculate the Jacobian for one link before doing the matrix version """
    radius = 3
    theta = 0.2

    # TODO: Create a 3D vector to the end point (r cos(theta), r sin(theta), 0)
    #   Needs to be 3D for cross product to work
# YOUR CODE HERE

    # The z vector we spin around
    omega_hat = [0, 0, 1]

    # TODO: Take the cross product of omega_hat and r
    #  This should always be 0 in 3rd component for a 2D vector
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

    # TODO: Solve the matrix using np.linalg.lstsq. Should return a 1x1 matrix with an angle change
    #   Note: Use rcond=None
    d_ang = np.zeros([1, 1])
# YOUR CODE HERE

    # Check result of solve - should be the same as dx_dy
    res = jacobian_matrix @ d_ang

    # The actual point you end up at if you change the angle by that much
    pt_new = [radius * np.cos(theta + d_ang), radius * np.sin(theta + d_ang)]

    print(f"J angle {res}, pt new {pt_new}")

    return d_ang


def calculate_jacobian(arm_with_angles):
    """
    Calculate the Jacobian from the angles and lengths in the arm
    Start with the wrist and work *backwards*, calculating Ri @ Ti @ previous_matrix
    The vector r from the practice Jacobian problem above is just the last column of that matrix
    @param arm_with_angles - The arm geometry, as constructed in arm_forward_kinematics, with angles
    @return 2xn Jacobian matrix that maps changes in angles to changes in x,y """

    # One column for each link plus the wrist
    jacob = np.zeros([2, len(arm_with_angles)])

    # TODO: To make things simpler, first build two lists that have the link lengths and angles in REVERSE order
    # Note that python has a handy reverse method for lists
    #  This is Python-ese for getting out the angles/lengths from the list of dictionaries
    #  Don't forget the wrist
    angles_links = [link["Angle"] for link in arm_with_angles[0:-1]]
    lengths_links = [link["Length"] for link in arm_with_angles[0:-1]]
    angles_reverse_order = []
    lengths_reverse_order = []
# YOUR CODE HERE

    # TODO: Now work backwards, calculating R @ T @ mat_accum
    #   In each iteration, update mat_accum THEN extract the
    #   vector r then do omega_hat cross r and put the result in the
    #   column of the matrix
    mat_accum = np.identity(3)
    # The z vector we spin around
    omega_hat = [0, 0, 1]

    # More python-ese - this gets each of the angles/lengths AND an enumeration variable i
    for i, (ang, length) in enumerate(zip(angles_reverse_order, lengths_reverse_order)):
        # TODO: Build mat_accum
        #       Get r from mat_accum
        #       Do omega_hat cross r
        #       Put the result in the n-i column in jacob - i.e., wrist should go in the last column in jacob
# YOUR CODE HERE
    return jacobian

def solve_jacobian(jacobian, vx_vy):

def jacobian(angles, arm, target, b_one_step=True):
    """
    Use jacobian to calculate move grasp point towards target
    @param angles - A list of angles for each link, followed by a triplet for the wrist and fingers
    @param arm - The arm geometry, as constructed in arm_forward_kinematics
    @param target - a 2x1 numpy array (x,y) that is the desired target point
    @param b_one_step - if True, return angles after one successful movement towards goal
    @ return angles that put the grasp point as close as possible to the target
    """

    """

        # begin homework 2 : Problem 2
        mats = self.robot_arm.get_matrices()
        jacob = np.zeros([2, 3])

        matrix_order = ['wrist', 'forearm', 'upperarm']
        mat_accum = np.identity(3)
        for i, c in enumerate(matrix_order):
            mat_accum = mats[c + '_R'] @ mats[c + '_T'] @ mat_accum
            r = [mat_accum[0, 2], mat_accum[1, 2], 0]
            omega_cross_r = np.cross(omega_hat, r)
            jacob[0:2, 2-i] = np.transpose(omega_cross_r[0:2])

        # Desired change in x,y
        pt_reach = self.robot_arm.arm_end_pt()
        dx_dy[0, 0] = self.reach_x.value() - pt_reach[0]
        dx_dy[1, 0] = self.reach_y.value() - pt_reach[1]

        # Use pseudo inverse to solve
        d_ang = np.linalg.lstsq(jacob, dx_dy, rcond=None)[0]
        res = jacob @ d_ang

        d_ang_save = [self.theta_slds[i].value() for i in range(0, 3)]
        d_min = 0
        v_min = pow(dx_dy[0, 0], 2) + pow(dx_dy[1, 0], 2)
        d_max = min(1, pi / max(d_ang))
        for i, ang in enumerate(d_ang):
            self.theta_slds[i].set_value(self.theta_slds[i].value() + ang)
        pt_reach_move = self.robot_arm.arm_end_pt()
        v_max = pow(pt_reach_move[0] - self.reach_x.value(), 2) + pow(pt_reach_move[1] - self.reach_y.value(), 2)
        d_try = d_min
        v_try = v_min
        if v_max < v_min:
            d_try = d_max
            v_try = v_max
        while d_max - d_min > 0.00001 and v_try > 0.01:
            d_try = 0.5 * (d_max + d_min)
            for i, ang in enumerate(d_ang):
                self.theta_slds[i].set_value(d_ang_save[i] + ang * d_try)
            pt_reach_try = self.robot_arm.arm_end_pt()
            v_try = pow(pt_reach_try[0] - self.reach_x.value(), 2) + pow(pt_reach_try[1] - self.reach_y.value(), 2)

            if v_try < v_min:
                v_min = v_try
                d_min = d_try
            elif v_try < v_max:
                v_max = v_try
                d_max = d_try
            elif v_max > v_min:
                v_max = v_try
                d_max = d_try
            else:
                v_min = v_try
                d_min = d_try

        if v_min < v_try and v_min < v_max:
            d_try = d_min

        if v_max < v_try and v_max < v_min:
            d_try = d_max

        for i, ang in enumerate(d_ang):
            self.theta_slds[i].set_value(d_ang_save[i] + ang * d_try)

        pt_reach_res = self.robot_arm.arm_end_pt()
        desired_text = "Desired dx dy {0:0.4f},{1:0.4f},".format(dx_dy[0, 0], dx_dy[1, 0])
        got_text = " got {0:0.4f},{1:0.4f}".format(res[0, 0], res[1, 0])
        actual_text = ", actual {0:0.4f},{1:0.4f}".format(pt_reach_res[0], pt_reach_res[1])
        self.robot_arm.text = desired_text + got_text + actual_text
        # to set text
        # self.robot_arm.text = text
        # end homework 2 problem 2
        """
