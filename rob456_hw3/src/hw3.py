#!/usr/bin/env python

import rospy
import math
import tf
from tf.transformations import euler_from_quaternion
import message_filters

# The laser scan message
from sensor_msgs.msg import LaserScan

# The odometry message
from nav_msgs.msg import Odometry

# the velocity command message
from geometry_msgs.msg import Twist

# instantiate global variables "globalOdom"
globalOdom = Odometry()

# global pi
pi = math.pi

# method to control the robot
def callback(scan,odom):
    # the odometry parameter should be global
    global globalOdom
    globalOdom = odom

    # make a new twist message
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
    
    # find current (x,y) position of robot based on odometry
    currentX = globalOdom.pose.pose.position.x
    currentY = globalOdom.pose.pose.position.y

    # find current orientation of robot based on odometry (quaternion coordinates)
    xOr = globalOdom.pose.pose.orientation.x
    yOr = globalOdom.pose.pose.orientation.y
    zOr = globalOdom.pose.pose.orientation.z
    wOr = globalOdom.pose.pose.orientation.w

    # find orientation of robot (Euler coordinates)
    (roll, pitch, yaw) = euler_from_quaternion([xOr, yOr, zOr, wOr])

    # find currentAngle of robot (equivalent to yaw)
    # now that you have yaw, the robot's pose is completely defined by (currentX, currentY, currentAngle)
    currentAngle = yaw

    # find laser scanner properties (min scan angle, max scan angle, scan angle increment)
    maxAngle = scan.angle_max
    minAngle = scan.angle_min
    angleIncrement = scan.angle_increment

    # find current laser angle, max scan length, distance array for all scans, and number of laser scans
    currentLaserTheta = minAngle
    maxScanLength = scan.range_max 
    distanceArray = scan.ranges
    numScans = len(distanceArray)
   
    # the code below (currently commented) shows how 
    # you can print variables to the terminal (may 
    # be useful for debugging)
    #print 'x: {0}'.format(currentX)
    #print 'y: {0}'.format(currentY)
    #print 'theta: {0}'.format(currentAngle)

    # for each laser scan
    distThreshold = 2.0   # obstacle avoidance threshold
    minScan = distanceArray[0]
    velocity = 5.0
    bearing = 0.0
    for curScan in range(0, numScans):
      if currentLaserTheta > -pi/2 and currentLaserTheta < 0 and distanceArray[curScan] < distThreshold:
        bearing = 1.0 # turn left
      elif currentLaserTheta >= 0 and currentLaserTheta < pi/2 and distanceArray[curScan] < distThreshold:
        bearing = -1.0 # turn right
      if distanceArray[curScan] < minScan:
        minScan = distanceArray[curScan]
      currentLaserTheta = currentLaserTheta + angleIncrement
    
    # set the robot motion
    # commanded velocities
    command.linear.x = velocity*min(1.0,minScan/distThreshold) # slow down if within threshold distance
    command.angular.z = bearing
    pub.publish(command)

# main function call
if __name__ == "__main__":
    # Initialize the node
    rospy.init_node('lab3', log_level=rospy.DEBUG)

    # subscribe to laser scan message
    sub = message_filters.Subscriber('base_scan', LaserScan)

    # subscribe to odometry message    
    sub2 = message_filters.Subscriber('odom', Odometry)

    # synchronize laser scan and odometry data
    ts = message_filters.TimeSynchronizer([sub, sub2], 10)
    ts.registerCallback(callback)

    # publish twist message
    pub = rospy.Publisher('cmd_vel',Twist)

    # Turn control over to ROS
    rospy.spin()

