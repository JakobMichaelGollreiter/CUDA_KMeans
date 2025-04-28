#include "kmeans.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <limits>
#include <algorithm>

// External functions from CUDA
extern "C" int assignClustersKernel(
    float* d_points,
    float* d_centroids,
    int* d_assignments,
    int* d_changes,
    int numPoints,
    int numClusters,
    int dimensions
);

extern "C" void updateCentroidsKernel(
    float* d_points,
    float* d_centroids,
    int* d_assignments,
    int* d_counts,
    int numPoints,
    int numClusters,
    int dimensions
);

extern "C" bool isCUDAAvailable();

// Constructor with parameters
KMeans::KMeans(int numClusters, int maxIter, double eps, bool gpu) 
    : k(numClusters), maxIterations(maxIter), epsilon(eps), useGPU(gpu),
      d_points(nullptr), d_centroids(nullptr), d_assignments(nullptr), 
      d_changes(nullptr), d_counts(nullptr), dimensions(0), numPoints(0) {
    
    // Check if GPU usage is requested but not available
    if (useGPU && !isCUDAAvailable()) {
        std::cerr << "Warning: CUDA is not available. Falling back to CPU implementation." << std::endl;
        useGPU = false;
    }
}

// Destructor
KMeans::~KMeans() {
    if (useGPU) {
        freeGPUMemory();
    }
}

// Calculate Euclidean distance between two points
double KMeans::distance(const std::vector<double>& a, const std::vector<double>& b) const {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Add a data point to the dataset
void KMeans::addPoint(const std::vector<double>& features) {
    points.emplace_back(features);
}

// Set centroids directly
void KMeans::setCentroids(const std::vector<std::vector<double>>& initialCentroids) {
    if (initialCentroids.size() != static_cast<size_t>(k)) {
        std::cerr << "Error: Number of provided centroids (" << initialCentroids.size() 
                  << ") doesn't match k (" << k << ")" << std::endl;
        return;
    }
    
    centroids = initialCentroids;
}

// Enable or disable GPU usage
void KMeans::setUseGPU(bool use) {
    if (use && !isCUDAAvailable()) {
        std::cerr << "Warning: CUDA is not available. Falling back to CPU implementation." << std::endl;
        useGPU = false;
    } else {
        useGPU = use;
    }
}

// Check if CUDA is available
bool KMeans::isCUDAAvailable() {
    return ::isCUDAAvailable();
}

// Load data points from CSV file
bool KMeans::loadDataFromCSV(const std::string& filename, char delimiter) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }
    
    points.clear();
    std::string line;
    
    // Skip header row
    std::getline(file, line);
    
    // Process each line in the CSV file
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<double> features;
        
        // Process each value in the line
        while (std::getline(ss, token, delimiter)) {
            try {
                double value = std::stod(token);
                features.push_back(value);
            } catch (const std::exception& e) {
                std::cerr << "Error: Could not convert '" << token << "' to double" << std::endl;
                return false;
            }
        }
        
        // Add point if we have valid features, but ignore the last column
        if (!features.empty()) {
            features.pop_back();  // Remove the last column
            addPoint(features);
        }
    }
    
    file.close();
    
    if (points.empty()) {
        std::cerr << "Error: No valid data points found in " << filename << std::endl;
        return false;
    }
    
    std::cout << "Successfully loaded " << points.size() << " data points from " 
              << filename << std::endl;
    return true;
}

// Load initial centroids from CSV file
bool KMeans::loadCentroidsFromCSV(const std::string& filename, char delimiter) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    std::vector<std::vector<double>> initialCentroids;
    std::string line;

    // Skip header row
    std::getline(file, line);

    // Process each line in the CSV file
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<double> centroid;

        // Process each value in the line
        while (std::getline(ss, token, delimiter)) {
            try {
                double value = std::stod(token);
                centroid.push_back(value);
            } catch (const std::exception& e) {
                std::cerr << "Error: Could not convert '" << token << "' to double" << std::endl;
                return false;
            }
        }

        // Add centroid if we have valid features
        if (!centroid.empty()) {
            initialCentroids.push_back(centroid);
        }
    }

    file.close();

    // Check if we have the right number of centroids
    if (initialCentroids.size() != static_cast<size_t>(k)) {
        std::cerr << "Error: Number of centroids in file (" << initialCentroids.size() 
                  << ") doesn't match k (" << k << ")" << std::endl;
        return false;
    }

    // Check if dimensions match
    if (!points.empty() && !initialCentroids.empty() && 
        points[0].features.size() != initialCentroids[0].size()) {
        std::cerr << "Error: Dimension mismatch between data points (" 
                  << points[0].features.size() << ") and centroids (" 
                  << initialCentroids[0].size() << ")" << std::endl;
        return false;
    }

    // Set the centroids
    setCentroids(initialCentroids);

    std::cout << "Successfully loaded " << initialCentroids.size() 
              << " centroids from " << filename << std::endl;
    return true;
}

// Allocate GPU memory
void KMeans::allocateGPUMemory() {
    if (!useGPU) return;
    
    numPoints = points.size();
    dimensions = points[0].features.size();
    
    // Allocate memory for points
    cudaMalloc(&d_points, numPoints * dimensions * sizeof(float));
    
    // Allocate memory for centroids
    cudaMalloc(&d_centroids, k * dimensions * sizeof(float));
    
    // Allocate memory for assignments
    cudaMalloc(&d_assignments, numPoints * sizeof(int));
    
    // Allocate memory for changes counter
    cudaMalloc(&d_changes, sizeof(int));
    
    // Allocate memory for cluster counts
    cudaMalloc(&d_counts, k * sizeof(int));
    
    // Initialize assignments to -1
    cudaMemset(d_assignments, -1, numPoints * sizeof(int));
}

// Free GPU memory
void KMeans::freeGPUMemory() {
    if (d_points) cudaFree(d_points);
    if (d_centroids) cudaFree(d_centroids);
    if (d_assignments) cudaFree(d_assignments);
    if (d_changes) cudaFree(d_changes);
    if (d_counts) cudaFree(d_counts);
    
    d_points = nullptr;
    d_centroids = nullptr;
    d_assignments = nullptr;
    d_changes = nullptr;
    d_counts = nullptr;
}

// Copy initial data to GPU - called once at the beginning
void KMeans::copyInitialDataToGPU() {
    if (!useGPU) return;
    
    // Copy points to device
    std::vector<float> h_points_flat(numPoints * dimensions);
    for (size_t i = 0; i < numPoints; i++) {
        for (size_t j = 0; j < dimensions; j++) {
            h_points_flat[i * dimensions + j] = static_cast<float>(points[i].features[j]);
        }
    }
    cudaMemcpy(d_points, h_points_flat.data(), numPoints * dimensions * sizeof(float), cudaMemcpyHostToDevice);
    
    // Copy centroids to device
    std::vector<float> h_centroids_flat(k * dimensions);
    for (int i = 0; i < k; i++) {
        for (size_t j = 0; j < dimensions; j++) {
            h_centroids_flat[i * dimensions + j] = static_cast<float>(centroids[i][j]);
        }
    }
    cudaMemcpy(d_centroids, h_centroids_flat.data(), k * dimensions * sizeof(float), cudaMemcpyHostToDevice);
}

// Copy final results from GPU - called once at the end
void KMeans::copyFinalResultsFromGPU() {
    if (!useGPU) return;
    
    // Copy assignments back to host
    std::vector<int> h_assignments(numPoints);
    cudaMemcpy(h_assignments.data(), d_assignments, numPoints * sizeof(int), cudaMemcpyDeviceToHost);
    
    for (size_t i = 0; i < numPoints; i++) {
        points[i].cluster = h_assignments[i];
    }
    
    // Copy centroids back to host
    std::vector<float> h_centroids_flat(k * dimensions);
    cudaMemcpy(h_centroids_flat.data(), d_centroids, k * dimensions * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < k; i++) {
        for (size_t j = 0; j < dimensions; j++) {
            centroids[i][j] = static_cast<double>(h_centroids_flat[i * dimensions + j]);
        }
    }
}

// Assign clusters - CPU version
int KMeans::assignClustersCPU() {
    int changes = 0;
    
    for (auto& point : points) {
        double minDist = std::numeric_limits<double>::max();
        int closestCluster = -1;
        
        for (int i = 0; i < k; i++) {
            // Check for dimension mismatch
            if (point.features.size() != centroids[i].size()) {
                std::cerr << "Error: Dimension mismatch between point (" 
                          << point.features.size() << ") and centroid " 
                          << i << " (" << centroids[i].size() << ")" << std::endl;
                continue;
            }
            
            double dist = distance(point.features, centroids[i]);
            if (dist < minDist) {
                minDist = dist;
                closestCluster = i;
            }
        }
        
        // Verify we found a valid cluster
        if (closestCluster == -1) {
            std::cerr << "Error: Could not assign point to any cluster" << std::endl;
            continue;
        }
        
        // Check if cluster assignment changed
        if (point.cluster != closestCluster) {
            point.cluster = closestCluster;
            changes++;
        }
    }
    
    return changes;
}

// Assign clusters - GPU version
int KMeans::assignClustersGPU() {
    // Call CUDA kernel - all processing stays on GPU
    return assignClustersKernel(
        d_points, d_centroids, d_assignments, d_changes,
        static_cast<int>(numPoints), k, static_cast<int>(dimensions)
    );
}

// Assign clusters - dispatcher
int KMeans::assignClusters() {
    if (useGPU) {
        return assignClustersGPU();
    } else {
        return assignClustersCPU();
    }
}

// Update centroids - CPU version
void KMeans::updateCentroidsCPU() {
    size_t dimensions = points[0].features.size();
    
    // Reset centroids to zero and count the number of points in each cluster
    std::vector<std::vector<double>> newCentroids(k, std::vector<double>(dimensions, 0.0));
    std::vector<int> counts(k, 0);
    
    // Sum all points in each cluster
    for (const auto& point : points) {
        int cluster = point.cluster;
        if (cluster < 0 || cluster >= k) {
            std::cerr << "Error: Invalid cluster assignment: " << cluster << std::endl;
            continue;
        }
        
        counts[cluster]++;
        
        for (size_t j = 0; j < dimensions; j++) {
            newCentroids[cluster][j] += point.features[j];
        }
    }
    
    // Calculate average (centroid) for each cluster
    for (int i = 0; i < k; i++) {
        // If cluster is empty, we need to handle it
        if (counts[i] == 0) {
            std::cout << "Warning: Cluster " << i << " is empty. Keeping old centroid." << std::endl;
            continue;
        }
        
        for (size_t j = 0; j < dimensions; j++) {
            newCentroids[i][j] /= counts[i];
        }
        
        // Update centroid
        centroids[i] = newCentroids[i];
    }
}

// Update centroids - GPU version
void KMeans::updateCentroidsGPU() {
    // Call CUDA kernel - all processing stays on GPU
    updateCentroidsKernel(
        d_points, d_centroids, d_assignments, d_counts,
        static_cast<int>(numPoints), k, static_cast<int>(dimensions)
    );
}

// Update centroids - dispatcher
void KMeans::updateCentroids() {
    if (useGPU) {
        updateCentroidsGPU();
    } else {
        updateCentroidsCPU();
    }
}

// Run the k-means algorithm
void KMeans::run() {
    if (points.empty()) {
        std::cerr << "Error: No data points loaded" << std::endl;
        return;
    }
    
    if (points.size() < static_cast<size_t>(k)) {
        std::cerr << "Error: Number of points (" << points.size() 
                  << ") must be at least equal to k (" << k << ")" << std::endl;
        return;
    }
    
    if (centroids.empty()) {
        std::cerr << "Error: Centroids not initialized. Please load centroids before running." << std::endl;
        return;
    }
    
    // Validate dimensions match
    size_t dim = points[0].features.size();
    for (const auto& centroid : centroids) {
        if (centroid.size() != dim) {
            std::cerr << "Error: Dimension mismatch between data points (" 
                      << dim << ") and centroids" << std::endl;
            return;
        }
    }
    
    // If using GPU, allocate memory and copy data once at the beginning
    if (useGPU) {
        std::cout << "Using GPU implementation for k-means clustering" << std::endl;
        allocateGPUMemory();
        copyInitialDataToGPU(); // Copy data to GPU only once
    } else {
        std::cout << "Using CPU implementation for k-means clustering" << std::endl;
    }
    
    // Main Lloyd's algorithm loop
    int iterations = 0;
    int changes;
    
    do {
        // Assign points to nearest centroids - all on GPU if useGPU is true
        changes = assignClusters();
        
        // Update centroids based on assigned points - all on GPU if useGPU is true
        updateCentroids();
        
        iterations++;
        
        std::cout << "Iteration " << iterations << ": " << changes << " points changed clusters" << std::endl;
        
    } while (changes > 0 && iterations < maxIterations);
    
    // If using GPU, copy results back once at the end
    if (useGPU) {
        copyFinalResultsFromGPU(); // Copy results back to host once at the end
        freeGPUMemory();  
    }
    
    std::cout << "K-means converged after " << iterations << " iterations." << std::endl;
}

// Get cluster assignments
std::vector<int> KMeans::getClusterAssignments() const {
    std::vector<int> assignments;
    for (const auto& point : points) {
        assignments.push_back(point.cluster);
    }
    return assignments;
}

// Get centroids
const std::vector<std::vector<double>>& KMeans::getCentroids() const {
    return centroids;
}

// Calculate the Sum of Squared Errors (SSE)
double KMeans::calculateSSE() const {
    double sse = 0.0;
    for (const auto& point : points) {
        // For SSE, we need the squared distance
        double dist = 0.0;
        for (size_t i = 0; i < point.features.size(); i++) {
            double diff = point.features[i] - centroids[point.cluster][i];
            dist += diff * diff;
        }
        sse += dist;  // Already the squared distance
    }
    return sse;
}

// Save cluster assignments to CSV file (only cluster column)
bool KMeans::saveClusterAssignmentsToCSV(const std::string& filename, char delimiter) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return false;
    }
    
    // Write header - only cluster
    file << "cluster" << std::endl;
    
    // Write each point's cluster assignment only
    for (const auto& point : points) {
        file << point.cluster << std::endl;
    }
    
    file.close();
    std::cout << "Successfully saved cluster assignments to " << filename << std::endl;
    return true;
}

// Save final centroids to CSV file
bool KMeans::saveCentroidsToCSV(const std::string& filename, char delimiter) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return false;
    }
    
    // Write header
    for (size_t i = 0; i < centroids[0].size(); i++) {
        file << "feature" << i;
        if (i < centroids[0].size() - 1) {
            file << delimiter;
        }
    }
    file << std::endl;
    
    // Write each centroid
    for (const auto& centroid : centroids) {
        for (size_t i = 0; i < centroid.size(); i++) {
            file << centroid[i];
            if (i < centroid.size() - 1) {
                file << delimiter;
            }
        }
        file << std::endl;
    }
    
    file.close();
    std::cout << "Successfully saved centroids to " << filename << std::endl;
    return true;
}
