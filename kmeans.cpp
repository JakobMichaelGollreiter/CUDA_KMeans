#include "kmeans.h"

// Constructor with parameters
KMeans::KMeans(int numClusters, int maxIter, double eps) 
    : k(numClusters), maxIterations(maxIter), epsilon(eps) {}

// Calculate Euclidean distance between two points
double KMeans::distance(const std::vector<double>& a, const std::vector<double>& b) const {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Add a data point to the dataset
void KMeans::addPoint(const std::vector<double>& features) {
    points.emplace_back(features);
}

// Initialize centroids using the k++ method for better initial placement
void KMeans::initializeCentroids() {
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
bool KMeans::assignClusters() {
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
double KMeans::updateCentroids() {
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

// Run the k-means algorithm
void KMeans::run() {
    if (points.size() < static_cast<size_t>(k)) {
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
std::vector<int> KMeans::getClusterAssignments() const {
    std::vector<int> assignments;
    for (const auto& point : points) {
        assignments.push_back(point.cluster);
    }
    return assignments;
}

// Get centroids
const std::vector<std::vector<double>>& KMeans::getCentroids() const {
    return centroids;
}

// Calculate the Sum of Squared Errors (SSE)
double KMeans::calculateSSE() const {
    double sse = 0.0;
    for (const auto& point : points) {
        sse += distance(point.features, centroids[point.cluster]);
    }
    return sse;
}