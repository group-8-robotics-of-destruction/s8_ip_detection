set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")

cmake_minimum_required(VERSION 2.8.3)
project(s8_ip_detection)

find_package(catkin REQUIRED COMPONENTS
  	sensor_msgs
	cv_bridge
	roscpp
	std_msgs
	image_transport
	message_generation
	pcl_conversions
  	pcl_ros
  	s8_common_node
)

#add_message_files(FILES
#	distPose.msg
#)

generate_messages(DEPENDENCIES
	std_msgs
)

catkin_package(
	DEPENDS
	CATKIN_DEPENDS message_runtime s8_common_node
)
include_directories(${catkin_INCLUDE_DIRS})


add_executable(object_detector src/object_detector.cpp)
target_link_libraries(object_detector ${catkin_LIBRARIES})