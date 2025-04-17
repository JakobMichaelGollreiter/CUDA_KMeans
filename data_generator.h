#ifndef DATA_GENERATOR_H
#define DATA_GENERATOR_H

#include <vector>
#include <random>
#include <iostream>

// Class for generating synthetic data for clustering
class DataGenerator {
private:
    std::mt19937 gen;  // Mersenne Twister random generator
    
public:
    // Constructor
    DataGenerator(unsigned seed = std::random_device{}());
    
    // Generate Gaussian clusters
    std::vector<std::vector<double>> generateGaussianClusters(
        const std::vector<std::vector<double>>& clusterCenters,
        const std::vector<double>& stdDevs,
        int pointsPerCluster
    );
    
    // Print information about generated clusters
    void printClusterInfo(
        const std::vector<std::vector<double>>& clusterCenters,
        const std::vector<double>& stdDevs
    );
};

#endif // DATA_GENERATOR_H