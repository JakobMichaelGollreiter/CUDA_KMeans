cmake_minimum_required(VERSION 3.10)
project(kmeans LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Find CUDA
find_package(CUDA QUIET)

# Set source files
set(SOURCES
    main.cpp
    kmeans.cpp
)

# Set CUDA source files
set(CUDA_SOURCES
    kmeans_kernels.cu
)

# Add executable
if(CUDA_FOUND)
    # Define executable with both C++ and CUDA sources
    add_executable(kmeans ${SOURCES} ${CUDA_SOURCES})
    
    # Enable separable compilation
    set_target_properties(kmeans PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    
    target_compile_definitions(kmeans PRIVATE CUDA_AVAILABLE)
    
    # Include directories
    target_include_directories(kmeans PRIVATE ${CUDA_INCLUDE_DIRS})
    
    # Link libraries
    target_link_libraries(kmeans ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
    
    message(STATUS "CUDA found. Building with GPU support.")
else()
    # Define executable with only C++ sources for CPU-only version
    add_executable(kmeans ${SOURCES})
    
    # Add a stub for the CUDA functions
    configure_file(
        ${CMAKE_SOURCE_DIR}/cuda_stub.cpp.in
        ${CMAKE_BINARY_DIR}/cuda_stub.cpp
        @ONLY
    )
    target_sources(kmeans PRIVATE ${CMAKE_BINARY_DIR}/cuda_stub.cpp)
    
    message(STATUS "CUDA not found. Building CPU-only version.")
endif()

# Create stub file if not exists
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/cuda_stub.cpp.in")
    file(WRITE "${CMAKE_SOURCE_DIR}/cuda_stub.cpp.in" "
// Stub implementations for CUDA functions
extern \"C\" int assignClustersKernel(
    float* d_points,
    float* d_centroids,
    int* d_assignments,
    int* d_changes,
    int numPoints,
    int numClusters,
    int dimensions
) {
    return 0;
}

extern \"C\" void updateCentroidsKernel(
    float* d_points,
    float* d_centroids,
    int* d_assignments,
    int* d_counts,
    int numPoints,
    int numClusters,
    int dimensions
) {
    // Do nothing in stub
}

extern \"C\" bool isCUDAAvailable() {
    return false;
}
")
endif()

# Install targets
install(TARGETS kmeans DESTINATION bin)