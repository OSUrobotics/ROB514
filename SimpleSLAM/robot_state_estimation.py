#!/usr/bin/env python3

import numpy as np

from world_state import WorldState
from door_sensor import DoorSensor


# Belief about world/robot state
class RobotStateEstimation:
    def __init__(self):

        # Probability representation (discrete)
        self.probabilities = []
        self.reset_probabilities(10)

        # Kalman (Gaussian) probabilities
        self.mean = 0.5
        self.standard_deviation = 0.4
        self.reset_kalman()

    def reset_probabilities(self, n_probability):
        """ Initialize discrete probability resolution with uniform distribution """
        div = 1.0 / n_probability
        self.probabilities = np.ones(n_probability) * div

    def update_belief_sensor_reading(self, ws, ds, sensor_reading_has_door):
        """ Update your probabilities based on the sensor reading being true (door) or false (no door)
        :param ws World state - has where the doors are
        :param ds Door Sensor - has probabilities for door sensor readings
        :param sensor_reading_has_door - contains true/false from door sensor
        """
        new_probs = np.zeros(len(self.probabilities))
        # begin homework 2 : problem 3
        # Normalize - all the denominators are the same because they're the sum of all cases
        # end homework 2 : problem 3

    # Distance to wall sensor (state estimation)
    def update_dist_sensor(self, ws, dist_reading):
        """ Update state estimation based on sensor reading
        :param ws - for standard deviation of wall sensor
        :param dist_reading - distance reading returned from the sensor, in range 0,1 (essentially, robot location) """

        # Standard deviation of error
        standard_deviation = ws.wall_standard_deviation
        # begin homework 2 : Extra credit
            # Sample from probability
        # Normalize - all the denominators are the same
        # end homework 2 : Extra credit
        return self.mean, self.standard_deviation

    def update_belief_move_left(self, rs):
        """ Update the probabilities assuming a move left.
        :param rs - robot state, has the probabilities"""

        # begin homework 2 problem 4
        # Check probability of left, no, right sum to one
        # Left edge - put move left probability into zero square along with stay-put probability
        # Right edge - put move right probability into last square
        # Normalize - sum should be one, except for numerical rounding
        # end homework 2 problem 4

    def update_belief_move_right(self, rs):
        """ Update the probabilities assuming a move right.
        :param rs - robot state, has the probabilities"""

        # begin homework 2 problem 4
        # Check probability of left, no, right sum to one
        # Left edge - put move left probability into zero square along with stay-put probability
        # Right edge - put move right probability into last square
        # Normalize - sum should be one, except for numerical rounding
        # end homework 2 problem 4

    # Put robot in the middle with a really broad standard deviation
    def reset_kalman(self):
        self.mean = 0.5
        self.standard_deviation = 0.4

    # Given a movement, update Gaussian
    def update_kalman_move(self, rs, amount):
        """ Kalman filter update mean/standard deviation with move (the prediction step)
        :param rs : robot state - has the standard deviation error for moving
        :param amount : The requested amount to move
        :return : mean and standard deviation of my current estimated location """

        # begin homework 3 : Problem 2
        # end homework 3 : Problem 2
        return self.mean, self.standard_deviation

    # Sensor reading, distance to wall (Kalman filtering)
    def update_gauss_sensor_reading(self, ws, dist_reading):
        """ Update state estimation based on sensor reading
        :param ws - for standard deviation of wall sensor
        :param dist_reading - distance reading returned"""

        # begin homework 3 : Problem 1
        # end homework 3 : Problem 1
        return self.mean, self.standard_deviation


if __name__ == '__main__':
    ws_global = WorldState()

    ds_global = DoorSensor()

    rse_global = RobotStateEstimation()

    # Check out these cases
    # We have two possibilities - either in front of door, or not - cross two sensor readings
    #   saw door versus not saw door
    uniform_prob = rse_global.probabilities[0]

    # begin homework 2 problem 4
    # Four cases - based on default door probabilities of
    # DoorSensor.prob_see_door_if_door = 0.8
    # DoorSensor.prob_see_door_if_no_door = 0.2
    #  and 10 probability divisions. Three doors visible.
    # probability saw door if door, saw door if no door, etc
    # Resulting probabilities, assuming 3 doors
    # Check that our probabilities are updated correctly
    # Spacing of bins
    # end homework 2 problem 4

    print("Passed tests")
