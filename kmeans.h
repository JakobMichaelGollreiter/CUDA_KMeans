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
class KMeans {
private:
    int k;              // Number of clusters
    int maxIterations;  // Maximum iterations before stopping
    double epsilon;     // Convergence threshold
    bool useGPU;        // Flag to indicate if GPU should be used

    // Helper struct for a point in N-dimensional space
    struct Point {
        std::vector<double> features;
        int cluster;

        Point(const std::vector<double>& f) : features(f), cluster(-1) {}
    };

    std::vector<Point> points;             // All data points
    std::vector<std::vector<double>> centroids;  // Cluster centroids
    
    // GPU specific members
    float* d_points;       // Device memory for points
    float* d_centroids;    // Device memory for centroids
    int* d_assignments;    // Device memory for cluster assignments
    int* d_changes;        // Device memory for tracking changes
    int* d_counts;         // Device memory for counts in each cluster
    size_t dimensions;     // Number of dimensions for points
    size_t numPoints;      // Number of points

    // Calculate Euclidean distance between two points
    double distance(const std::vector<double>& a, const std::vector<double>& b) const;

    // Assign each point to nearest centroid - CPU version
    // Returns the number of points that changed clusters
    int assignClustersCPU();
    
    // Assign each point to nearest centroid - GPU version
    // Returns the number of points that changed clusters
    int assignClustersGPU();
    
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
    
    // Copy data to GPU
    void copyToGPU();
    
    // Copy results from GPU
    void copyFromGPU();

public:
    KMeans(int numClusters, int maxIter = 100, double eps = 1e-4, bool gpu = false);
    ~KMeans();
    
    // Add a data point to the dataset
    void addPoint(const std::vector<double>& features);
    
    // Set centroids directly
    void setCentroids(const std::vector<std::vector<double>>& initialCentroids);
    
    // Load data points from CSV file
    bool loadDataFromCSV(const std::string& filename, char delimiter = ',');
    
    // Load initial centroids from CSV file
    bool loadCentroidsFromCSV(const std::string& filename, char delimiter = ',');
    
    // Run the k-means algorithm
    void run();
    
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
    
    // Check if CUDA is available
    static bool isCUDAAvailable();
};

#endif // KMEANS_H