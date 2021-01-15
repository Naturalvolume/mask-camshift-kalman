cmake_minimum_required(VERSION 2.8)
project( tracking )
set(OpenCV_DIR /opt/ros/kinetic/share/OpenCV-3.3.1-dev)
set(OpenCV_FOUND 1)
find_package(OpenCV REQUIRED)
add_executable(tracking main.cpp tracker.cpp tracker.hpp functions.hpp)
target_link_libraries(tracking ${OpenCV_LIBS})


