#!/usr/bin/env python
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


def draw_points(points, pub):
    """
    Plots an array of points [(x, y)...] in rviz

    :param: points iterable of (x, y) pairs. If a numpy array, shape should be (n, 2)
    :return: None
    """
    msg = Marker()
    # Marker header specifies what (and when) it is drawn relative to
    msg.header.frame_id = "map"
    msg.header.stamp = rospy.Time.now()
    # uint8 POINTS=8
    msg.type = 8
    # Disappear after 1sec. Comment this line out to make them persist indefinitely
    msg.lifetime = rospy.rostime.Duration(1, 0)
    # Set marker visual properties
    msg.color.b = 1.0
    msg.color.a = 1.0
    msg.scale.x = 0.2
    msg.scale.y = 0.2
    # Copy  (x, y) into message and publish
    for (x, y) in points:
        p = Point()
        p.x = x
        p.y = y
        p.z = 0.1 # Places all points 10cm above the ground
        msg.points.append(p)
    pub.publish(msg)


if __name__ == '__main__':
    pub = rospy.Publisher('/marker', Marker, queue_size=2)
    rospy.init_node('rviz_demo')
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        draw_points([(1, 2), (2, 3), (3, 4)], pub)
        rate.sleep()
