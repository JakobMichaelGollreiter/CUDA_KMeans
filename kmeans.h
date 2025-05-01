#ifndef KMEANS_H
#define KMEANS_H

#include <vector>
#include <iostream>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <limits>
#include <algorithm>
#include <cuda_runtime.h>

// Class for K-means clustering implementation using Lloyd's algorithm
// Optimized to keep data on GPU throughout the entire algorithm
// Using Structure of Arrays (SoA) layout for better memory coalescing
class KMeans {
private:
    int k;              // Number of clusters
    int maxIterations;  // Maximum iterations before stopping
    double epsilon;     // Convergence threshold
    bool useGPU;        // Flag to indicate if GPU should be used
    bool useTriangleInequality; // Flag to use triangle inequality optimization

    // Helper struct for a point in N-dimensional space (for CPU implementation)
    struct Point {
        std::vector<double> features;
        int cluster;
        double distToAssignedCentroid; // Distance to currently assigned centroid

        Point(const std::vector<double>& f) : features(f), cluster(-1), distToAssignedCentroid(std::numeric_limits<double>::max()) {}
    };

    std::vector<Point> points;             // All data points (host)
    std::vector<std::vector<double>> centroids;  // Cluster centroids (host)
    std::vector<std::vector<double>> centroidDistances; // Distances between centroids
    
    // GPU specific members
    float* d_points_soa;   // Device memory for points in SoA format
    float* d_centroids;    // Device memory for centroids in SoA format
    float* d_centroidDistances; // Device memory for centroid distances
    int* d_assignments;    // Device memory for cluster assignments
    int* d_changes;        // Device memory for tracking changes
    int* d_counts;         // Device memory for counts in each cluster
    float* d_pointCentroidDists; // Device memory for point-to-centroid distances
    size_t dimensions;     // Number of dimensions for points
    size_t numPoints;      // Number of points

    // Calculate Euclidean distance between two points (CPU version)
    double distance(const std::vector<double>& a, const std::vector<double>& b) const;

    // Calculate distances between all centroids
    void calculateCentroidDistances();

    // Update centroid distances on GPU
    void calculateCentroidDistancesGPU();

    // Assign each point to nearest centroid - CPU version
    int assignClustersCPU();
    
    // Assign each point to nearest centroid - CPU version with triangle inequality
    int assignClustersCPUWithTriangleInequality();
    
    // Assign each point to nearest centroid - GPU version
    int assignClustersGPU();
    
    // Assign each point to nearest centroid - GPU version with triangle inequality
    int assignClustersGPUWithTriangleInequality();
    
    // Combined function that calls either CPU or GPU version
    int assignClusters();

    // Update centroids based on current cluster assignments - CPU version
    void updateCentroidsCPU();
    
    // Update centroids based on current cluster assignments - GPU version
    void updateCentroidsGPU();
    
    // Combined function that calls either CPU or GPU version
    void updateCentroids();
    
    // Allocate GPU memory
    void allocateGPUMemory();
    
    // Free GPU memory
    void freeGPUMemory();
    
    // Copy initial data to GPU (called once)
    void copyInitialDataToGPU();
    
    // Copy results from GPU (called once at the end)
    void copyFinalResultsFromGPU();
    
    // Debug methods
    void verifyDataTransfer();
    void verifyAssignments(int iteration);
    
    // Warmup functions for GPU kernels
    void warmupGPUKernels();
    void warmupAssignClustersKernel();
    void warmupUpdateCentroidsKernel();
    void warmupTriangleInequalityKernels();

public:
    KMeans(int numClusters, int maxIter = 100, double eps = 1e-4, bool gpu = false, bool useTriangle = false);
    ~KMeans();
    
    // Add a data point to the dataset
    void addPoint(const std::vector<double>& features);
    
    // Set centroids directly
    void setCentroids(const std::vector<std::vector<double>>& initialCentroids);
    
    // Load data points from CSV file
    bool loadDataFromCSV(const std::string& filename, char delimiter = ',');
    
    // Load initial centroids from CSV file
    bool loadCentroidsFromCSV(const std::string& filename, char delimiter = ',');
    
    // Original run method (calls all stages)
    void run();
    
    // NEW: Prepare GPU memory (allocate and transfer) - called before timing
    double prepareGPUMemory();
    
    // NEW: Run just the algorithm (Lloyd's iterations) - this is timed
    void runAlgorithm();
    
    // NEW: Retrieve results from GPU - called after timing
    void retrieveResultsFromGPU();
    
    // Get cluster assignments
    std::vector<int> getClusterAssignments() const;
    
    // Get centroids
    const std::vector<std::vector<double>>& getCentroids() const;
    
    // Calculate the Sum of Squared Errors (SSE)
    double calculateSSE() const;
    
    // Save cluster assignments to CSV file
    bool saveClusterAssignmentsToCSV(const std::string& filename, char delimiter = ',') const;
    
    // Save final centroids to CSV file
    bool saveCentroidsToCSV(const std::string& filename, char delimiter = ',') const;
    
    // Enable or disable GPU usage
    void setUseGPU(bool use);
    
    // Enable or disable triangle inequality optimization
    void setUseTriangleInequality(bool use);
    
    // Check if CUDA is available
    static bool isCUDAAvailable();
    
    // Public method to run warmup before timing
    void warmupKernels();
};

#endif // KMEANS_H
