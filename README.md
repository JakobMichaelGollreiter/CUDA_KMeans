# K-means Clustering Implementation
This project implements Lloyd's algorithm for K-means clustering in C++. It includes both a core clustering algorithm and utilities for generating synthetic test data with reproducible results.
## Overview 
K-means clustering is a popular unsupervised machine learning algorithm that partitions data points into K distinct clusters based on distance to the nearest cluster centroid. This implementation uses Lloyd's algorithm with the k-means++ initialization method for improved initial centroid placement.
## Features
- Lloyd's Algorithm: Efficient implementation of the standard k-means algorithm
- K-means++ Initialization: Smart initial centroid selection to improve convergence
- Reproducible Results: Seed-based random number generation for consistent outcomes
- Dimensionality Independent: Works with any number of dimensions/features
- Synthetic Data Generation: Gaussian cluster generator for testing
- Performance Metrics: Calculates Sum of Squared Errors (SSE) for measuring clustering quality
## Project Structure
The project is organized into the following files:
- kmeans.h and kmeans.cpp: Core K-means clustering implementation
- data_generator.h and data_generator.cpp: Synthetic data generation utilities
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
Run the program with an optional seed parameter for reproducible results:
bash./build/kmeans_serial [seed]


