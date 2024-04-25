

cmake_minimum_required(VERSION 2.5)

project(get_ventricle_delneation)

# Find ITK
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(get_ventricle_delineation ./source/get_ventricle_delineation.cpp)

target_link_libraries(get_ventricle_delineation ${ITK_LIBRARIES})