#!/usr/bin/env python
import rospy
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Bool

class GlobalPlannar(object):
    def __init__(self):
        self.map_data = None
        self.odom_pos = None
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.pub = rospy.Publisher('/recieved_map', Bool, queue_size=10)

    def map_callback(self, msg):
        rospy.loginfo('In map callback')
        self.map_data = msg.data
        self.pub.publish(True)

    def odom_callback(self, msg):
        rospy.loginfo('In odom callback' + str(self.odom_pos))
        self.odom_pos = msg.pose.pose

if __name__ == '__main__':
    rospy.init_node('class_example')
    gp = GlobalPlannar()
    rospy.spin()

