

cmake_minimum_required(VERSION 2.5)

project(relabeling_8VOIs)

# Find ITK
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(relabeling_8VOIs ./source/relabeling_8VOIs.cpp)

target_link_libraries(relabeling_8VOIs ${ITK_LIBRARIES})