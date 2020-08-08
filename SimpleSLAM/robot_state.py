#!/usr/bin/env python3

import numpy as np


from world_state import WorldState


# Door sensor - needs to now the world state to answer questions
class RobotState:
    def __init__(self):
        # Default probabilities - will be set later by gui
        self.set_move_left_probabilities(0.8, 0.1)
        self.set_move_right_probabilities(0.8, 0.1)

        self.prob_move_left_if_left = 0.8
        self.prob_move_right_if_left = 0.05
        self.prob_no_move_if_left = 0.15

        self.prob_move_left_if_right = 0.8
        self.prob_move_right_if_right = 0.05
        self.prob_no_move_if_right = 0.15

        self.robot_move_SD_err = 0.0

        # Gaussian probabilities
        self.set_move_gauss_probabilities(0.05)

        # Where the robot actually is
        self.robot_loc = 0.5

    # Make sure probabilities add up to one
    def set_move_left_probabilities(self, move_left_if_left=0.8, move_right_if_left=0.05):
        self.prob_move_left_if_left = move_left_if_left
        self.prob_move_right_if_left = move_right_if_left

        # begin homework 2 : problem 2
        # check probabilities are correct
        # end homework 2 : problem 2

    # Make sure probabilities add up to one
    def set_move_right_probabilities(self, move_right_if_right=0.8, move_left_if_right=0.05):
        self.prob_move_right_if_right = move_right_if_right
        self.prob_move_left_if_right = move_left_if_right

        # begin homework 2 : problem 2
        # check probabilities are correct
        # end homework 2 : problem 2

    # Just a helper function to place robot + sign in middle of bin
    def adjust_location(self, n_divs):
        div = 1.0 / n_divs
        bin_id = min(n_divs-1, max(0, np.round(self.robot_loc / div)))
        self.robot_loc = (bin_id + 0.5) * div

    # For Kalman filtering - error in movement
    def set_move_gauss_probabilities(self, standard_deviation):
        self.robot_move_SD_err = standard_deviation

    # Actually move - don't move off of end of hallway
    def _move_(self, amount):
        """ Move the amount given - but clamp value
        :param amount - requested amount to move
        :return amount - amount actually moved"""
        if 0 <= self.robot_loc + amount <= 1:
            self.robot_loc += amount

        return amount

    # Roll the dice and move
    def move_left(self, step_size):
        """ Move to the left by step_size (probably)
        :param step_size - the bin size
        :returns The amount actually moved """
        # begin homework 2 : problem 2
        # begin homework 2 : problem 2
        # Flip the coin...
        # Determine whether to move left, right, or stay put
        # end homework 2 : problem 2
        return self._move_(step_size)

    # Roll the dice and move
    def move_right(self, step_size):
        """ Move to the right by step_size (probably)
        :param step_size - the bin size
        :returns The amount actually moved """
        # begin homework 2 : problem 2
        # Flip the coin...
        # Determine whether to move left, right, or stay put
        # end homework 2 : problem 2
        return self._move_(step_size)

    def move_gauss(self, amount):
        """ Move by the amount given (may be positive or negative) with noise added
        :param amount - amount to move (negative is left, positive right)
        :returns The amount actually moved """
        # begin homework 3 : problem 2
        # Sample the noise distribution - note zero mean
        # Move amount plus noise sampled from noise distribution
        # end homework 3 : problem 2

        # Actually move (don't run off of end)
        return self._move_(amount)


if __name__ == '__main__':
    ws = WorldState()

    rs = RobotState()

    # Move far enough to the left and you should stop moving
    print("Checking _Move_ function")
    check_step_size = 0.1
    for i in range(0, 20):
        rs.move_left(check_step_size)
        if rs.robot_loc < 0 or rs.robot_loc > 1:
            raise ValueError("Robot went off end of left wall")

    # Repeat for right
    for i in range(0, 20):
        rs.move_right(check_step_size)
        if rs.robot_loc < 0 or rs.robot_loc > 1:
            raise ValueError("Robot went off end of right wall")

    # Check that we get our probabilites back (mostly)
    print("Checking move left probabilities")
    count_moved_left = 0
    count_moved_right = 0
    for i in range(0, 1000):
        rs.robot_loc = 0.5
        rs.move_left(check_step_size)
        if rs.robot_loc == 0.5 - check_step_size:
            count_moved_left += 1
        elif rs.robot_loc == 0.5 + check_step_size:
            count_moved_right += 1

    prob_count_left = count_moved_left/1000
    prob_count_right = count_moved_right/1000
    if abs(prob_count_left - rs.prob_move_left_if_left) > 0.1:
        raise ValueError("Probability should be close to {}, is {}".format(rs.prob_move_left_if_left, prob_count_left))
    if abs(prob_count_right - rs.prob_move_right_if_left) > 0.1:
        raise ValueError("Probability should be close to {}, is {}".format(rs.prob_move_right_if_left, prob_count_right))

    print("Checking move right probabilities")
    count_moved_left = 0
    count_moved_right = 0
    for i in range(0, 1000):
        rs.robot_loc = 0.5
        rs.move_right(check_step_size)
        if rs.robot_loc == 0.5 - check_step_size:
            count_moved_left += 1
        elif rs.robot_loc == 0.5 + check_step_size:
            count_moved_right += 1

    prob_count_left = count_moved_left / 1000
    prob_count_right = count_moved_right / 1000
    if abs(prob_count_left - rs.prob_move_left_if_right) > 0.1:
        raise ValueError("Probability should be close to {}, is {}".format(rs.prob_move_left_if_right, prob_count_left))
    if abs(prob_count_right - rs.prob_move_right_if_right) > 0.1:
        raise ValueError("Probability should be close to {}, is {}".format(rs.prob_move_right_if_right, prob_count_right))

    print("Checking move with normal distribution probabilities")
    dist_moved = []
    rs.set_move_gauss_probabilities(0.1)
    for i in range(0, 1000):
        rs.robot_loc = 0.5

        dist_moved.append(rs.move_gauss(0))

    mu_moved = np.mean(dist_moved)
    SD_moved = np.std(dist_moved)
    if abs(mu_moved) > 0.01:
        raise ValueError("Mean should be close to 0, is {}".format(mu_moved))
    if abs(SD_moved - rs.robot_move_SD_err) > 0.01:
        raise ValueError("Standard deviation should be close to {}, is {}".format(rs.robot_move_SD_err, SD_moved))

    print("Passed tests")
