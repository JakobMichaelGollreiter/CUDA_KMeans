cmake_minimum_required(VERSION 3.14)
project(kmeans_clustering)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Serial implementation
add_executable(kmeans_serial 
    main.cpp 
    kmeans.cpp 
    kmeans.h
)

# Set output directory
set_target_properties(kmeans_serial
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)