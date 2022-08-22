#!/usr/bin/env python

import rospy

# Sensor message types
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

# the velocity command message
from geometry_msgs.msg import Twist

GOAL = (1, 1)
ODOM = None

def lidar_callback(scan_msg):
    global GOAL, ODOM
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

    print ODOM
    # Problem 1: move the robot toward the goal
    # YOUR CODE HERE
    # End problem 1

    currentLaserTheta = minAngle
    # for each laser scan
    for i, scan in enumerate(distances):
        # for each laser scan, the angle is currentLaserTheta, the index is i, and the distance is scan
        # Problem 2: avoid obstacles based on laser scan readings
        # TODO YOUR CODE HERE
        # End problem 2
        # After this loop is done, we increment the currentLaserTheta
        currentLaserTheta = currentLaserTheta + angleIncrement

    pub.publish(command)


def odom_callback(msg):
    """
    Subscribes to the odom message, unpacks and transforms the relevent information, and places it in the global variable ODOM
    ODOM is structured as follows:
    ODOM = (x, y, yaw)

    :param: msg: Odometry message
    :returns: None
    """
    global ODOM
    position = msg.pose.pose.position
    ori = msg.pose.pose.orientation
    (r, p, yaw) = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
    ODOM = (position.x, position.y, yaw)


if __name__ == "__main__":
    # Initialize the node
    rospy.init_node('lab2', log_level=rospy.DEBUG)

    # subscribe to sensor messages
    lidar_sub = rospy.Subscriber('/scan', LaserScan, lidar_callback)
    odom_sub = rospy.Subscriber('/odom', Odometry, odom_callback)

    # publish twist message
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    # Turn control over to ROS
    rospy.spin()
