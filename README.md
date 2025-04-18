# K-means Clustering Implementation
This project implements Lloyd's algorithm for K-means clustering in C++. It includes both a core clustering algorithm and utilities for generating synthetic test data with reproducible results.
## Project Structure
The project is organized into the following files:
- kmeans.h and kmeans.cpp: Core K-means clustering implementation
- main.cpp: Example program demonstrating usage
- CMakeLists.txt: CMake build configuration

## Building the Project 
### Prerequisites
- C++11 compatible compiler
- CMake (version 3.14 or higher)
### Build Instructions
- Clone the repository
Create a build directory and run CMake: mkdir -p cmake_build; cd cmake_build; cmake ..

- Build the project: make
## Usage
This implementation takes in data, initial centroid, number of clusters, and number of iterations as arguments.
./build/kmeans_serial ../example_data.csv ../example_init.csv 3 100


