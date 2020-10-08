cmake_minimum_required(VERSION 2.8.3)
project(rob456_hw3)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  gmapping
  nav_msgs
  roscpp
  rospy
  stage_ros
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)
