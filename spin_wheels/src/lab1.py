#!/usr/bin/env python

import rospy

# Sensor message types
from sensor_msgs.msg import LaserScan

# the velocity command message
from geometry_msgs.msg import Twist


def lidar_callback(scan_msg):
    # Let's make a new twist message
    command = Twist()

    # Fill in the fields.  Field values are unspecified 
    # until they are actually assigned. The Twist message 
    # holds linear and angular velocities.
    command.linear.x = 0.0
    command.linear.y = 0.0
    command.linear.z = 0.0
    command.angular.x = 0.0
    command.angular.y = 0.0
    command.angular.z = 0.0

    # Lidar properties (unpacked for your ease of use)
    # find current laser angle, max scan length, distance array for all scans, and number of laser scans
    maxAngle = scan_msg.angle_max
    minAngle = scan_msg.angle_min
    angleIncrement = scan_msg.angle_increment

    maxScanLength = scan_msg.range_max
    distances = scan_msg.ranges
    numScans = len(distances)

    currentLaserTheta = minAngle
    # for each laser scan
    for i, scan in enumerate(distances):
        # for each laser scan, the angle is currentLaserTheta, the index is i, and the distance is scan
        # TODO YOUR CODE HERE
        # based on the motion you want (turn, stop moving, etc...), modify the target velocity of the robot motion
        # i.e.:
        # command.linear.x = 0.0
        # command.angular.z = 0.0
        # After this loop is done, we increment the currentLaserTheta
        currentLaserTheta = currentLaserTheta + angleIncrement

    pub.publish(command)


if __name__ == "__main__":
    # Initialize the node
    rospy.init_node('lab1', log_level=rospy.DEBUG)

    # subscribe to lidar laser scan message
    lidar_sub = rospy.Subscriber('/scan', LaserScan, lidar_callback)

    # publish twist message
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    # Turn control over to ROS
    rospy.spin()
