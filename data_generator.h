#ifndef DATA_GENERATOR_H
#define DATA_GENERATOR_H

#include <vector>
#include <random>
#include <iostream>

// Class for generating synthetic data for clustering
class DataGenerator {
private:
    std::mt19937 gen;  // Mersenne Twister random generator
    unsigned int seed;  // Stored seed value for reproducibility
    
public:
    // Constructor with optional seed (default is random)
    DataGenerator(unsigned int seed = std::random_device{}());
    
    // Get the current seed value
    unsigned int getSeed() const;
    
    // Reset the generator with a new seed
    void setSeed(unsigned int newSeed);
    
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