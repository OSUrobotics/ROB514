#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
import tf2_geometry_msgs
from geometry_msgs.msg import Pose, PoseStamped, Quaternion, Point, PointStamped
from std_msgs.msg import Header
import tf2_ros

class PoseListener(object):
    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.pose_pub = rospy.Publisher('/map_odom', Pose, queue_size=10)

    def odom_callback(self, msg):
        try:
            h = Header()
            h.stamp = rospy.get_rostime()
            h.frame_id = 'base_footprint'
            pose = PoseStamped(h, Pose(Point(0,0,0), Quaternion(0,0,0,1)))
            
            new_pose = self.tf_buffer.transform(pose, 'map', rospy.Duration(1.0))
            rospy.logdebug(new_pose)
            self.pose_pub.publish(new_pose.pose)

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(e)
            return

if __name__ == '__main__':
    rospy.init_node('map_odom_tf')
    pl = PoseListener()
    rospy.spin()

