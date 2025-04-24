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

// Set centroids directly
void KMeans::setCentroids(const std::vector<std::vector<double>>& initialCentroids) {
    if (initialCentroids.size() != static_cast<size_t>(k)) {
        std::cerr << "Error: Number of provided centroids (" << initialCentroids.size() 
                  << ") doesn't match k (" << k << ")" << std::endl;
        return;
    }
    
    centroids = initialCentroids;
}

// Load data points from CSV file
bool KMeans::loadDataFromCSV(const std::string& filename, char delimiter) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }
    
    points.clear();
    std::string line;
    
    // Skip header row
    std::getline(file, line);
    
    // Process each line in the CSV file
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<double> features;
        
        // Process each value in the line
        while (std::getline(ss, token, delimiter)) {
            try {
                double value = std::stod(token);
                features.push_back(value);
            } catch (const std::exception& e) {
                std::cerr << "Error: Could not convert '" << token << "' to double" << std::endl;
                return false;
            }
        }
        
        // Add point if we have valid features, but ignore the last column
        if (!features.empty()) {
            features.pop_back();  // Remove the last column
            addPoint(features);
        }
    }
    
    file.close();
    
    if (points.empty()) {
        std::cerr << "Error: No valid data points found in " << filename << std::endl;
        return false;
    }
    
    std::cout << "Successfully loaded " << points.size() << " data points from " 
              << filename << std::endl;
    return true;
}

// Load initial centroids from CSV file
// Load initial centroids from CSV file
bool KMeans::loadCentroidsFromCSV(const std::string& filename, char delimiter) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    std::vector<std::vector<double>> initialCentroids;
    std::string line;

    // Skip header row
    std::getline(file, line);

    // Process each line in the CSV file
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<double> centroid;

        // Process each value in the line
        while (std::getline(ss, token, delimiter)) {
            try {
                double value = std::stod(token);
                centroid.push_back(value);
            } catch (const std::exception& e) {
                std::cerr << "Error: Could not convert '" << token << "' to double" << std::endl;
                return false;
            }
        }

        // Add centroid if we have valid features
        if (!centroid.empty()) {
            initialCentroids.push_back(centroid);
        }
    }

    file.close();

    // Check if we have the right number of centroids
    if (initialCentroids.size() != static_cast<size_t>(k)) {
        std::cerr << "Error: Number of centroids in file (" << initialCentroids.size() 
                  << ") doesn't match k (" << k << ")" << std::endl;
        return false;
    }

    // Check if dimensions match
    if (!points.empty() && !initialCentroids.empty() && 
        points[0].features.size() != initialCentroids[0].size()) {
        std::cerr << "Error: Dimension mismatch between data points (" 
                  << points[0].features.size() << ") and centroids (" 
                  << initialCentroids[0].size() << ")" << std::endl;
        return false;
    }

    // Set the centroids
    setCentroids(initialCentroids);

    std::cout << "Successfully loaded " << initialCentroids.size() 
              << " centroids from " << filename << std::endl;
    return true;
}

// Assign each point to nearest centroid
int KMeans::assignClusters() {
    int changes = 0;
    
    for (auto& point : points) {
        double minDist = std::numeric_limits<double>::max();
        int closestCluster = -1;
        
        for (int i = 0; i < k; i++) {
            // Check for dimension mismatch
            if (point.features.size() != centroids[i].size()) {
                std::cerr << "Error: Dimension mismatch between point (" 
                          << point.features.size() << ") and centroid " 
                          << i << " (" << centroids[i].size() << ")" << std::endl;
                continue;
            }
            
            double dist = distance(point.features, centroids[i]);
            if (dist < minDist) {
                minDist = dist;
                closestCluster = i;
            }
        }
        
        // Verify we found a valid cluster
        if (closestCluster == -1) {
            std::cerr << "Error: Could not assign point to any cluster" << std::endl;
            continue;
        }
        
        // Check if cluster assignment changed
        if (point.cluster != closestCluster) {
            point.cluster = closestCluster;
            changes++;
        }
    }
    
    return changes;
}

// Update centroids based on current cluster assignments
void KMeans::updateCentroids() {
    size_t dimensions = points[0].features.size();
    
    // Reset centroids to zero and count the number of points in each cluster
    std::vector<std::vector<double>> newCentroids(k, std::vector<double>(dimensions, 0.0));
    std::vector<int> counts(k, 0);
    
    // Sum all points in each cluster
    for (const auto& point : points) {
        int cluster = point.cluster;
        if (cluster < 0 || cluster >= k) {
            std::cerr << "Error: Invalid cluster assignment: " << cluster << std::endl;
            continue;
        }
        
        counts[cluster]++;
        
        for (size_t j = 0; j < dimensions; j++) {
            newCentroids[cluster][j] += point.features[j];
        }
    }
    
    // Calculate average (centroid) for each cluster
    for (int i = 0; i < k; i++) {
        // If cluster is empty, we need to handle it
        if (counts[i] == 0) {
            std::cout << "Warning: Cluster " << i << " is empty. Keeping old centroid." << std::endl;
            continue;
        }
        
        for (size_t j = 0; j < dimensions; j++) {
            newCentroids[i][j] /= counts[i];
        }
        
        // Update centroid
        centroids[i] = newCentroids[i];
    }
}

// Run the k-means algorithm
void KMeans::run() {
    if (points.empty()) {
        std::cerr << "Error: No data points loaded" << std::endl;
        return;
    }
    
    if (points.size() < static_cast<size_t>(k)) {
        std::cerr << "Error: Number of points (" << points.size() 
                  << ") must be at least equal to k (" << k << ")" << std::endl;
        return;
    }
    
    if (centroids.empty()) {
        std::cerr << "Error: Centroids not initialized. Please load centroids before running." << std::endl;
        return;
    }
    
    // Validate dimensions match
    size_t dim = points[0].features.size();
    for (const auto& centroid : centroids) {
        if (centroid.size() != dim) {
            std::cerr << "Error: Dimension mismatch between data points (" 
                      << dim << ") and centroids" << std::endl;
            return;
        }
    }
    
    // Main Lloyd's algorithm loop
    int iterations = 0;
    int changes;
    
    do {
        // Assign points to nearest centroids
        changes = assignClusters();
        
        // Update centroids based on assigned points
        updateCentroids();
        
        iterations++;
        
        std::cout << "Iteration " << iterations << ": " << changes << " points changed clusters" << std::endl;
        
    } while (changes > 0 && iterations < maxIterations);
    
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
        // The distance function already returns Euclidean distance
        // For SSE, we need the squared distance
        double dist = 0.0;
        for (size_t i = 0; i < point.features.size(); i++) {
            double diff = point.features[i] - centroids[point.cluster][i];
            dist += diff * diff;
        }
        sse += dist;  // Already the squared distance
    }
    return sse;
}

// Save cluster assignments to CSV file (only cluster column)
bool KMeans::saveClusterAssignmentsToCSV(const std::string& filename, char delimiter) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return false;
    }
    
    // Write header - only cluster
    file << "cluster" << std::endl;
    
    // Write each point's cluster assignment only
    for (const auto& point : points) {
        file << point.cluster << std::endl;
    }
    
    file.close();
    std::cout << "Successfully saved cluster assignments to " << filename << std::endl;
    return true;
}

// Save final centroids to CSV file
bool KMeans::saveCentroidsToCSV(const std::string& filename, char delimiter) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return false;
    }
    
    // Write header
    for (size_t i = 0; i < centroids[0].size(); i++) {
        file << "feature" << i;
        if (i < centroids[0].size() - 1) {
            file << delimiter;
        }
    }
    file << std::endl;
    
    // Write each centroid
    for (const auto& centroid : centroids) {
        for (size_t i = 0; i < centroid.size(); i++) {
            file << centroid[i];
            if (i < centroid.size() - 1) {
                file << delimiter;
            }
        }
        file << std::endl;
    }
    
    file.close();
    std::cout << "Successfully saved centroids to " << filename << std::endl;
    return true;
}