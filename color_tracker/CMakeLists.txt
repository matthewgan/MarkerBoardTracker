cmake_minimum_required(VERSION 2.8)
project (color_tracker)
set(CMAKE_MODULE_PATH ${CMAKE_INSTALL_PREFIX}/lib/cmake/ )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(color_tracker color_tracker.cpp)
target_link_libraries(color_tracker ${OpenCV_LIBS} )
