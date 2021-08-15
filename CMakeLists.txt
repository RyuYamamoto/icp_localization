cmake_minimum_required(VERSION 3.0.2)
project(icp_localization)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  ndt_omp
  roscpp
  sensor_msgs
  std_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
  tf2_sensor_msgs
  pcl_ros
)

catkin_package(
  INCLUDE_DIRS include
#  LIBRARIES icp_localization
  CATKIN_DEPENDS geometry_msgs roscpp sensor_msgs std_msgs tf2 tf2_geometry_msgs tf2_ros tf2_sensor_msgs pcl_ros
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(icp_localization_node src/icp_localization_node.cpp src/icp_localization.cpp)
target_link_libraries(icp_localization_node ${catkin_LIBRARIES})
