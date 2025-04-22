#ifndef KMEANS_H
#define KMEANS_H

#include <vector>
#include <iostream>
#include <cmath>
#include <random>
#include <limits>
#include <algorithm>

// Class for K-means clustering implementation using Lloyd's algorithm
class KMeans {
private:
    int k;              // Number of clusters
    int maxIterations;  // Maximum iterations before stopping
    double epsilon;     // Convergence threshold

    // Helper struct for a point in N-dimensional space
    struct Point {
        std::vector<double> features;
        int cluster;

        Point(const std::vector<double>& f) : features(f), cluster(-1) {}
    };

    std::vector<Point> points;             // All data points
    std::vector<std::vector<double>> centroids;  // Cluster centroids

    // Calculate Euclidean distance between two points
    double distance(const std::vector<double>& a, const std::vector<double>& b) const;

    // Initialize centroids using the k++ method for better initial placement
    void initializeCentroids();

    // Assign each point to nearest centroid
    bool assignClusters();

    // Update centroids based on current cluster assignments
    double updateCentroids();

public:
    KMeans(int numClusters, int maxIter = 300, double eps = 1e-4);
    
    // Add a data point to the dataset
    void addPoint(const std::vector<double>& features);
    
    // Run the k-means algorithm
    void run();
    
    // Get cluster assignments
    std::vector<int> getClusterAssignments() const;
    
    // Get centroids
    const std::vector<std::vector<double>>& getCentroids() const;
    
    // Calculate the Sum of Squared Errors (SSE)
    double calculateSSE() const;
};

#endif // KMEANS_H