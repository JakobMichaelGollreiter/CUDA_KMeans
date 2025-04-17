cmake_minimum_required(VERSION 3.14)
project(kmeans_clustering)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Find OpenMP package for potential future parallel implementation
find_package(OpenMP)

# Serial implementation
add_executable(kmeans_serial 
    main.cpp 
    kmeans.cpp 
    data_generator.cpp 
    kmeans.h 
    data_generator.h
)

# Set output directory
set_target_properties(kmeans_serial
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/build"
)

# Configure job scripts if needed
# configure_file(job-kmeans-serial job-kmeans-serial COPYONLY)

# Optional: Create a parallel OpenMP version 
if(OpenMP_CXX_FOUND)
    add_executable(kmeans_parallel
        main.cpp
        kmeans_parallel.cpp  # You'll need to create this file
        data_generator.cpp
        kmeans.h
        data_generator.h
    )
    
    target_link_libraries(kmeans_parallel PRIVATE OpenMP::OpenMP_CXX)
    
    set_target_properties(kmeans_parallel
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/build"
    )
    
    # configure_file(job-kmeans-parallel job-kmeans-parallel COPYONLY)
endif()
