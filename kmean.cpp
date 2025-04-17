#include <iostream>
#include <vector>
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
    double distance(const std::vector<double>& a, const std::vector<double>& b) const {
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    // Initialize centroids using the k++ method for better initial placement
    void initializeCentroids() {
        // Choose first centroid randomly
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(0, points.size() - 1);
        
        centroids.clear();
        centroids.push_back(points[distrib(gen)].features);
        
        // Choose remaining centroids with probability proportional to distance
        while (centroids.size() < static_cast<size_t>(k)) {
            std::vector<double> distances(points.size());
            double sum = 0.0;
            
            for (size_t i = 0; i < points.size(); i++) {
                // Find minimum distance to any existing centroid
                double minDist = std::numeric_limits<double>::max();
                for (const auto& centroid : centroids) {
                    double d = distance(points[i].features, centroid);
                    minDist = std::min(minDist, d);
                }
                distances[i] = minDist * minDist;  // Square for weighting
                sum += distances[i];
            }
            
            // Convert distances to probabilities
            for (size_t i = 0; i < distances.size(); i++) {
                distances[i] /= sum;
            }
            
            // Select next centroid based on probability distribution
            std::discrete_distribution<> dist(distances.begin(), distances.end());
            centroids.push_back(points[dist(gen)].features);
        }
    }

    // Assign each point to nearest centroid
    bool assignClusters() {
        bool changed = false;
        
        for (auto& point : points) {
            double minDist = std::numeric_limits<double>::max();
            int closestCluster = -1;
            
            for (int i = 0; i < k; i++) {
                double dist = distance(point.features, centroids[i]);
                if (dist < minDist) {
                    minDist = dist;
                    closestCluster = i;
                }
            }
            
            // Check if cluster assignment changed
            if (point.cluster != closestCluster) {
                point.cluster = closestCluster;
                changed = true;
            }
        }
        
        return changed;
    }

    // Update centroids based on current cluster assignments
    double updateCentroids() {
        std::vector<std::vector<double>> newCentroids(k, std::vector<double>(points[0].features.size(), 0.0));
        std::vector<int> counts(k, 0);
        
        // Sum all points in each cluster
        for (const auto& point : points) {
            int cluster = point.cluster;
            counts[cluster]++;
            
            for (size_t j = 0; j < point.features.size(); j++) {
                newCentroids[cluster][j] += point.features[j];
            }
        }
        
        // Calculate average (centroid) for each cluster
        for (int i = 0; i < k; i++) {
            // If cluster is empty, keep old centroid
            if (counts[i] == 0) continue;
            
            for (size_t j = 0; j < newCentroids[i].size(); j++) {
                newCentroids[i][j] /= counts[i];
            }
        }
        
        // Calculate total centroid movement
        double totalMovement = 0.0;
        for (int i = 0; i < k; i++) {
            totalMovement += distance(centroids[i], newCentroids[i]);
            centroids[i] = newCentroids[i];
        }
        
        return totalMovement;
    }

public:
    KMeans(int numClusters, int maxIter = 100, double eps = 1e-4) 
        : k(numClusters), maxIterations(maxIter), epsilon(eps) {}
    
    // Add a data point to the dataset
    void addPoint(const std::vector<double>& features) {
        points.emplace_back(features);
    }
    
    // Run the k-means algorithm
    void run() {
        if (points.size() < k) {
            std::cerr << "Error: Number of points must be at least equal to k" << std::endl;
            return;
        }
        
        // Initialize centroids
        initializeCentroids();
        
        // Main Lloyd's algorithm loop
        bool changed = true;
        double movement = std::numeric_limits<double>::max();
        int iterations = 0;
        
        while (changed && movement > epsilon && iterations < maxIterations) {
            // Assignment step: assign each point to nearest centroid
            changed = assignClusters();
            
            // Update step: recalculate centroids
            movement = updateCentroids();
            
            iterations++;
        }
        
        std::cout << "K-means converged after " << iterations << " iterations." << std::endl;
    }
    
    // Get cluster assignments
    std::vector<int> getClusterAssignments() const {
        std::vector<int> assignments;
        for (const auto& point : points) {
            assignments.push_back(point.cluster);
        }
        return assignments;
    }
    
    // Get centroids
    const std::vector<std::vector<double>>& getCentroids() const {
        return centroids;
    }
    
    // Calculate the Sum of Squared Errors (SSE)
    double calculateSSE() const {
        double sse = 0.0;
        for (const auto& point : points) {
            sse += distance(point.features, centroids[point.cluster]);
        }
        return sse;
    }
};

// Example usage
int main() {
    // Random number generation setup
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Number of clusters and dimensions
    int numClusters = 3;
    int dimensions = 2;
    int pointsPerCluster = 10;  
    
    // Create a KMeans instance
    KMeans kmeans(numClusters);
    
    // Define cluster centers (means for Gaussian distributions)
    std::vector<std::vector<double>> clusterCenters = {
        {2.0, 2.0},    // Cluster 1 center
        {8.0, 8.0},    // Cluster 2 center
        {-5.0, -5.0}     // Cluster 4 center
    };
    
    // Standard deviations for each cluster
    std::vector<double> stdDevs = {0.1, 0.2, 0.8};
    
    // Store all generated points for display later
    std::vector<std::vector<double>> dataPoints;
    
    // Generate points from Gaussian distributions for each cluster
    std::cout << "Generating " << pointsPerCluster << " points for each of " << numClusters << " clusters...\n";
    
    for (int c = 0; c < numClusters; c++) {
        std::cout << "Cluster " << c << " true center: (";
        for (size_t d = 0; d < clusterCenters[c].size(); d++) {
            std::cout << clusterCenters[c][d];
            if (d < clusterCenters[c].size() - 1) std::cout << ", ";
        }
        std::cout << "), StdDev: " << stdDevs[c] << std::endl;
        
        // Create normal distributions for this cluster
        std::vector<std::normal_distribution<double>> dists;
        for (int d = 0; d < dimensions; d++) {
            dists.emplace_back(clusterCenters[c][d], stdDevs[c]);
        }
        
        // Generate points for this cluster
        for (int i = 0; i < pointsPerCluster; i++) {
            std::vector<double> point;
            for (int d = 0; d < dimensions; d++) {
                point.push_back(dists[d](gen));
            }
            dataPoints.push_back(point);
            kmeans.addPoint(point);
        }
    }
    
    // Run the algorithm
    std::cout << "\nRunning K-means clustering...\n" << std::endl;
    kmeans.run();
    
    // Get cluster assignments and centroids
    auto assignments = kmeans.getClusterAssignments();
    auto centroids = kmeans.getCentroids();
    
    // Print each cluster's centroid
    std::cout << "\nEstimated cluster centroids from K-means:" << std::endl;
    std::cout << "-----------------" << std::endl;
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
    for (int i = 0; i < numClusters; i++) {
        std::cout << "Cluster " << i << ": " << clusterCounts[i] << " points" << std::endl;
    }
    
    // Print first 5 points from each cluster as a sample
    std::cout << "\nSample points from each cluster:" << std::endl;
    for (size_t c = 0; c < centroids.size(); c++) {
        std::cout << "Cluster " << c << " samples:" << std::endl;
        
        int count = 0;
        for (size_t i = 0; i < assignments.size(); i++) {
            if (assignments[i] == c) {
                std::cout << "  Point (";
                for (size_t j = 0; j < dataPoints[i].size(); j++) {
                    std::cout << dataPoints[i][j];
                    if (j < dataPoints[i].size() - 1) std::cout << ", ";
                }
                std::cout << ")" << std::endl;
                
                count++;
            }
        }
    }
    
    std::cout << "\nSum of Squared Errors: " << kmeans.calculateSSE() << std::endl;
    
    return 0;
}