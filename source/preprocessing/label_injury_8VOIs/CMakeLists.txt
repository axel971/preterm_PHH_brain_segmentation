

cmake_minimum_required(VERSION 2.5)

project(label_injury_8VOIs)

# Find ITK
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(label_injury_8VOIs ./source/label_injury_8VOIs.cpp)

target_link_libraries(label_injury_8VOIs ${ITK_LIBRARIES})