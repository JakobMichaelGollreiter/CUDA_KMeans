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

# Set source files
set(SOURCES
    main.cpp
    kmeans.cpp
)

# Set CUDA source files
set(CUDA_SOURCES
    kmeans_kernels.cu
)

# Add executable with both C++ and CUDA sources
add_executable(kmeans ${SOURCES} ${CUDA_SOURCES})

# Enable separable compilation
set_target_properties(kmeans PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link CUDA libraries manually
find_package(CUDAToolkit REQUIRED)

target_link_libraries(kmeans PRIVATE CUDA::cudart CUDA::cublas)

# Install target
install(TARGETS kmeans DESTINATION bin)
