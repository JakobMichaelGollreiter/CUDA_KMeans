#include "kmeans.h"
#include "data_generator.h"
#include <vector>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    // Parameters for clustering
    int numClusters = 3;
    int pointsPerCluster = 10;
    
    // Seed for reproducibility (default: 12345)
    unsigned int seed = 12345;
    
    // Allow seed to be specified as command line argument
    if (argc > 1) {
        try {
            seed = std::stoi(argv[1]);
        } catch (const std::exception& e) {
            std::cerr << "Invalid seed parameter. Using default: " << seed << std::endl;
        }
    }
    
    // Create a KMeans instance
    KMeans kmeans(numClusters);
    
    // Define cluster centers (means for Gaussian distributions)
    std::vector<std::vector<double>> clusterCenters = {
        {2.0, 2.0},     // Cluster 0 center
        {8.0, 8.0},     // Cluster 1 center
        {-5.0, -5.0}    // Cluster 2 center
    };
    
    // Standard deviations for each cluster
    std::vector<double> stdDevs = {0.1, 0.2, 0.8};
    
    // Initialize data generator with specified seed
    DataGenerator generator(seed);
    
    // Print information about the clusters we're generating
    std::cout << "Generating " << pointsPerCluster << " points for each of " 
              << numClusters << " clusters with seed " << seed << "...\n";
    generator.printClusterInfo(clusterCenters, stdDevs);
    
    // Generate synthetic data points
    std::vector<std::vector<double>> dataPoints = 
        generator.generateGaussianClusters(clusterCenters, stdDevs, pointsPerCluster);
    
    // Add all generated points to the kmeans object
    for (const auto& point : dataPoints) {
        kmeans.addPoint(point);
    }
    
    // Run the clustering algorithm
    std::cout << "\nRunning K-means clustering...\n" << std::endl;
    kmeans.run();
    
    // Get cluster assignments and centroids
    auto assignments = kmeans.getClusterAssignments();
    auto centroids = kmeans.getCentroids();
    
    // Print each cluster's centroid
    std::cout << "\nEstimated cluster centroids from K-means:" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
    for (size_t i = 0; i < centroids.size(); i++) {
        std::cout << "Centroid " << i << ": (";
        for (size_t j = 0; j < centroids[i].size(); j++) {
            std::cout << centroids[i][j];
            if (j < centroids[i].size() - 1) std::cout << ", ";
        }
        std::cout << ")" << std::endl;
    }
    
    // Count points in each cluster
    std::vector<int> clusterCounts(numClusters, 0);
    for (int cluster : assignments) {
        clusterCounts[cluster]++;
    }
    
    // Print cluster statistics
    std::cout << "\nCluster point counts:" << std::endl;
    std::cout << "---------------------" << std::endl;
    for (int i = 0; i < numClusters; i++) {
        std::cout << "Cluster " << i << ": " << clusterCounts[i] << " points" << std::endl;
    }
    
    // Print sample points from each cluster
    std::cout << "\nSample points from each cluster:" << std::endl;
    std::cout << "-------------------------------" << std::endl;
    for (size_t c = 0; c < centroids.size(); c++) {
        std::cout << "Cluster " << c << " samples:" << std::endl;
        
        int count = 0;
        for (size_t i = 0; i < assignments.size(); i++) {
            if (assignments[i] == static_cast<int>(c)) {
                std::cout << "  Point (";
                for (size_t j = 0; j < dataPoints[i].size(); j++) {
                    std::cout << dataPoints[i][j];
                    if (j < dataPoints[i].size() - 1) std::cout << ", ";
                }
                std::cout << ")" << std::endl;
                
                count++;
                // if (count >= 5) break;  // Show only first 5 points per cluster
            }
        }
    }
    
    std::cout << "\nSum of Squared Errors: " << kmeans.calculateSSE() << std::endl;
    
    return 0;
}