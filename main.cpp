#include "kmeans.h"
#include <vector>
#include <iostream>
#include <string>
#include <stdexcept>

int main(int argc, char* argv[]) {
    // Check command line arguments
    if (argc < 3 || argc > 5) {
        std::cerr << "Usage: " << argv[0] << " <data_file.csv> <centroids_file.csv> <num_clusters> [max_iterations]" << std::endl;
        std::cerr << "  <data_file.csv>    : Path to CSV file containing data points" << std::endl;
        std::cerr << "  <centroids_file.csv>: Path to CSV file containing initial centroids" << std::endl;
        std::cerr << "  <num_clusters>     : Number of clusters (k)" << std::endl;
        std::cerr << "  [max_iterations]   : Maximum iterations (default: 100)" << std::endl;
        return 1;
    }
    
    try {
        // Parse command line arguments
        std::string dataFile = argv[1];
        std::string centroidsFile = argv[2];
        
        int numClusters;
        try {
            numClusters = std::stoi(argv[3]);
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid number of clusters. Must be a positive integer." << std::endl;
            return 1;
        }
        
        if (numClusters <= 0) {
            std::cerr << "Error: Number of clusters must be positive." << std::endl;
            return 1;
        }
        
        // Optional argument
        int maxIterations = 100;  // Default value
        if (argc >= 5) {
            try {
                maxIterations = std::stoi(argv[4]);
                if (maxIterations <= 0) {
                    std::cerr << "Error: Maximum iterations must be positive. Using default: 100" << std::endl;
                    maxIterations = 100;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error: Invalid maximum iterations. Using default: 100" << std::endl;
            }
        }
        
        // Create a KMeans instance
        KMeans kmeans(numClusters, maxIterations);
        
        // Load data points from CSV file
        std::cout << "Loading data points from " << dataFile << "..." << std::endl;
        if (!kmeans.loadDataFromCSV(dataFile)) {
            std::cerr << "Failed to load data points." << std::endl;
            return 1;
        }
        
        // Load initial centroids from CSV file
        std::cout << "Loading initial centroids from " << centroidsFile << "..." << std::endl;
        if (!kmeans.loadCentroidsFromCSV(centroidsFile)) {
            std::cerr << "Failed to load centroids." << std::endl;
            return 1;
        }
        
        // Run the clustering algorithm
        std::cout << "\nRunning K-means clustering..." << std::endl;
        kmeans.run();
        
        // Get cluster assignments and centroids
        auto assignments = kmeans.getClusterAssignments();
        auto centroids = kmeans.getCentroids();
        
        // Print each cluster's centroid
        std::cout << "\nFinal cluster centroids:" << std::endl;
        std::cout << "-----------------------" << std::endl;
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
        
        std::cout << "\nSum of Squared Errors (SSE): " << kmeans.calculateSSE() << std::endl;
        
        // Save results to CSV files
        std::string outputClustersFile = dataFile + ".clusters.csv";
        std::string outputCentroidsFile = dataFile + ".centroids.csv";
        
        kmeans.saveClusterAssignmentsToCSV(outputClustersFile);
        kmeans.saveCentroidsToCSV(outputCentroidsFile);
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}