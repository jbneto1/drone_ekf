cmake_minimum_required(VERSION 3.0.2)
project(drone_ekf)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  mavros_msgs
  diagnostic_msgs
  tf2_ros
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/ekf_node.py
  scripts/plotter.py
  scripts/controller_frame_diagnostics.py
  scripts/offline_controller_frame_probe.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
