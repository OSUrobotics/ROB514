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


        # Use the text field to say what happened
        self.robot_arm.text = "Not improved"

        # begin homework 2 : Problem 1
        b_improved = False
        d_scl = 0.1
        d_eps = pi/10000
        # Keep trying smaller increments while nothing improves
        while d_scl > 0.0001 and not b_improved:
            # calculate the current distance
            pt = self.robot_arm.arm_end_pt()
            dist = pow(pt[0] - self.reach_x.value(), 2) + pow(pt[1] - self.reach_y.value(), 2)
            # Try each angle in turn
            for ang in self.theta_slds:
                save_ang = ang.value()

                # Gradient
                ang.set_value(save_ang - d_eps)
                pt_new = self.robot_arm.arm_end_pt()
                dist_new = pow(pt_new[0] - self.reach_x.value(), 2) + pow(pt_new[1] - self.reach_y.value(), 2)

                ang_try = save_ang + d_scl * pi
                if dist_new < dist:
                    ang_try = save_ang - 0.99 * d_scl * pi

                ang.set_value(ang_try)
                pt_new = self.robot_arm.arm_end_pt()
                dist_new = pow(pt_new[0] - self.reach_x.value(), 2) + pow(pt_new[1] - self.reach_y.value(), 2)
                if dist_new < dist:
                    b_improved = True
                    dist = dist_new
                    self.robot_arm.text = "Improved {} eps {}".format(ang.name, d_scl)
                else:
                    ang.set_value(save_ang)
            d_scl = d_scl / 2
        # end homework 2 : Problem 1




        # An example problem of an arm with radius 3 currently at angle theta
        radius = 3
        theta = 0.2
        # Vector to the end point
        r = [radius * cos(theta), radius * sin(theta), 0]
        # Spin around z
        omega_hat = [0, 0, 1]
        # always 0 in 3rd component
        omega_cross_r = np.cross(omega_hat, r)
        # Desired x,y change
        dx_dy = np.zeros([2, 1])
        dx_dy[0, 0] = -0.01
        dx_dy[1, 0] = -0.1
        # Jacobian
        J = np.zeros([2, 1])
        J[0:2, 0] = np.transpose(omega_cross_r[0:2])
        # Solve
        d_ang = np.linalg.lstsq(J, dx_dy, rcond=None)[0]
        # Check result of solve - should be the same as dx_dy
        res = J @ d_ang
        # The actual point you end up at if you change the angle by that much
        pt_new = [radius * cos(theta + d_ang), radius * sin(theta + d_ang)]

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
