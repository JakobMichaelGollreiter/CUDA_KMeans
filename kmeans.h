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

    // Assign each point to nearest centroid
    // Returns the number of points that changed clusters
    int assignClusters();

    // Update centroids based on current cluster assignments
    void updateCentroids();

public:
    KMeans(int numClusters, int maxIter = 100, double eps = 1e-4);
    
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
};

#endif // KMEANS_H