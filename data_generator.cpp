#include "data_generator.h"

// Constructor with optional seed
DataGenerator::DataGenerator(unsigned int seed) : seed(seed), gen(seed) {
    std::cout << "Data generator initialized with seed: " << seed << std::endl;
}

// Get the current seed value
unsigned int DataGenerator::getSeed() const {
    return seed;
}

// Reset the generator with a new seed
void DataGenerator::setSeed(unsigned int newSeed) {
    seed = newSeed;
    gen.seed(seed);
    std::cout << "Data generator reset with new seed: " << seed << std::endl;
}

// Generate Gaussian clusters
std::vector<std::vector<double>> DataGenerator::generateGaussianClusters(
    const std::vector<std::vector<double>>& clusterCenters,
    const std::vector<double>& stdDevs,
    int pointsPerCluster
) {
    std::vector<std::vector<double>> dataPoints;
    int numClusters = clusterCenters.size();
    int dimensions = clusterCenters[0].size();
    
    std::cout << "Generating points with seed: " << seed << std::endl;
    
    // Generate points for each cluster
    for (int c = 0; c < numClusters; c++) {
        // Create normal distributions for this cluster (one per dimension)
        std::vector<std::normal_distribution<double>> dists;
        for (int d = 0; d < dimensions; d++) {
            dists.emplace_back(clusterCenters[c][d], stdDevs[c]);
        }
        
        // Generate pointsPerCluster points for this cluster
        for (int i = 0; i < pointsPerCluster; i++) {
            std::vector<double> point;
            for (int d = 0; d < dimensions; d++) {
                point.push_back(dists[d](gen));
            }
            dataPoints.push_back(point);
        }
    }
    
    return dataPoints;
}

// Print information about generated clusters
void DataGenerator::printClusterInfo(
    const std::vector<std::vector<double>>& clusterCenters,
    const std::vector<double>& stdDevs
) {
    int numClusters = clusterCenters.size();
    
    for (int c = 0; c < numClusters; c++) {
        std::cout << "Cluster " << c << " true center: (";
        for (size_t d = 0; d < clusterCenters[c].size(); d++) {
            std::cout << clusterCenters[c][d];
            if (d < clusterCenters[c].size() - 1) std::cout << ", ";
        }
        std::cout << "), StdDev: " << stdDevs[c] << std::endl;
    }
}