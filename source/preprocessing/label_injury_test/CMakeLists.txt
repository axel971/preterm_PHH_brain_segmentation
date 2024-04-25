

cmake_minimum_required(VERSION 2.5)

project(label_injury)

# Find ITK
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(label_injury ./source/label_injury.cpp)

target_link_libraries(label_injury ${ITK_LIBRARIES})