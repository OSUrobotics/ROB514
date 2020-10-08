#!/usr/bin/env python

import rospy
import math
import tf
from tf.transformations import euler_from_quaternion
import message_filters

# The odometry message
from nav_msgs.msg import Odometry

def callback(truth,odom):
    # Calculate error between truth and odometry estimate
    x_ODOM_error = truth.pose.pose.position.x - odom.pose.pose.position.x
    y_ODOM_error = truth.pose.pose.position.y - odom.pose.pose.position.y
    
    xo = odom.pose.pose.orientation.x
    yo = odom.pose.pose.orientation.y
    zo = odom.pose.pose.orientation.z
    wo = odom.pose.pose.orientation.w
    
    xt = truth.pose.pose.orientation.x
    yt = truth.pose.pose.orientation.y
    zt = truth.pose.pose.orientation.z
    wt = truth.pose.pose.orientation.w

    # find orientation of robot (Euler coordinates)
    (ro, po, yo) = euler_from_quaternion([xo, yo, zo, wo])    
    (rt, pt, yt) = euler_from_quaternion([xt, yt, zt, wt])  
    t_ODOM_error = yt-yo  
    
    print "{0},{1},{2},{3}".format(rospy.get_time(),x_ODOM_error, y_ODOM_error, t_ODOM_error)


# main function call
if __name__ == "__main__":
    # Initialize the node
    rospy.init_node('odom_error_printer')

    # subscribe to truth pose message
    sub_truth = message_filters.Subscriber('base_pose_ground_truth', Odometry)

    # subscribe to odometry message    
    sub_odom = message_filters.Subscriber('odom', Odometry)

    # synchronize truth pose and odometry data
    ts = message_filters.TimeSynchronizer([sub_truth, sub_odom], 10)
    ts.registerCallback(callback)

    # Turn control over to ROS
    rospy.spin()
