cmake_minimum_required(VERSION 2.8)
project (marker_tracker)
set(CMAKE_MODULE_PATH ${CMAKE_INSTALL_PREFIX}/lib/cmake/ )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(marker_tracker marker_tracker.cpp)
target_link_libraries(marker_tracker ${OpenCV_LIBS} )
