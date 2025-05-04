#include "kmeans.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <limits>
#include <algorithm>
#include <cstddef>
#include <cuda_runtime.h>

// External functions from CUDA
extern "C" int assignClustersKernel(
    float* d_points_soa,
    float* d_centroids,
    int* d_assignments,
    int* d_changes,
    int numPoints,
    int numClusters,
    int dimensions,
    int startIdx = 0
);

extern "C" int assignClustersWithTriangleInequalityKernel(
    float* d_points_soa,
    float* d_centroids,
    float* d_centroidDistances,
    float* d_pointCentroidDists,
    int* d_assignments,
    int* d_changes,
    int numPoints,
    int numClusters,
    int dimensions,
    int startIdx = 0
);

extern "C" void updateCentroidsKernel(
    float* d_points_soa,
    float* d_centroids,
    int* d_assignments,
    int* d_counts,
    int numPoints,
    int numClusters,
    int dimensions,
    int startIdx = 0
);

extern "C" void calculateCentroidDistancesKernel(
    float* d_centroids, 
    float* d_centroidDistances,
    int numClusters, 
    int dimensions
);

extern "C" void accumulateBatchCentroidsKernel(
    float* d_batchCentroids,
    int* d_batchCounts,
    float* d_accumulatedCentroids,
    int* d_accumulatedCounts,
    int numClusters,
    int dimensions
);

extern "C" void finalizeCentroidsKernel(
    float* centroids,
    const int* counts,
    int numClusters,
    int dimensions,
    int gridSizeX,
    int gridSizeY,
    int blockSizeX,
    int blockSizeY
);

extern "C" bool isCUDAAvailable();

extern "C" void getAvailableGPUMemory(size_t* free, size_t* total);

// Constructor with parameters
KMeans::KMeans(int numClusters, int maxIter, double eps, bool gpu, bool useTriangle) 
    : k(numClusters), maxIterations(maxIter), epsilon(eps), useGPU(gpu), useTriangleInequality(useTriangle),
      batchSize(0), d_points_soa(nullptr), d_centroids(nullptr), d_centroidDistances(nullptr), 
      d_assignments(nullptr), d_changes(nullptr), d_counts(nullptr), d_pointCentroidDists(nullptr),
      d_accumulated_centroids(nullptr), d_accumulated_counts(nullptr), dimensions(0), numPoints(0) {
    
    // Check if GPU usage is requested but not available
    if (useGPU && !isCUDAAvailable()) {
        std::cerr << "Warning: CUDA is not available. Falling back to CPU implementation." << std::endl;
        useGPU = false;
    }
}

// Destructor
KMeans::~KMeans() {
    if (useGPU) {
        freeGPUMemory();
    }
}

// Calculate Euclidean distance between two points
double KMeans::distance(const std::vector<double>& a, const std::vector<double>& b) const {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Calculate distances between all centroids
void KMeans::calculateCentroidDistances() {
    centroidDistances.resize(k, std::vector<double>(k, 0.0));
    
    for (int i = 0; i < k; i++) {
        for (int j = i + 1; j < k; j++) {
            double dist = distance(centroids[i], centroids[j]);
            centroidDistances[i][j] = dist;
            centroidDistances[j][i] = dist; // Symmetric
        }
    }
}

// Calculate centroid distances on GPU
void KMeans::calculateCentroidDistancesGPU() {
    calculateCentroidDistancesKernel(
        d_centroids, 
        d_centroidDistances, 
        k, 
        static_cast<int>(dimensions)
    );
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
    
    // If using triangle inequality, calculate centroid distances
    if (useTriangleInequality && !centroids.empty()) {
        calculateCentroidDistances();
    }
}

// Enable or disable GPU usage
void KMeans::setUseGPU(bool use) {
    if (use && !isCUDAAvailable()) {
        std::cerr << "Warning: CUDA is not available. Falling back to CPU implementation." << std::endl;
        useGPU = false;
    } else {
        useGPU = use;
    }
}

// Enable or disable triangle inequality
void KMeans::setUseTriangleInequality(bool use) {
    useTriangleInequality = use;
}

// Set batch size manually
void KMeans::setBatchSize(int size) {
    batchSize = size;
}

// Check if CUDA is available
bool KMeans::isCUDAAvailable() {
    return ::isCUDAAvailable();
}

// Estimate memory requirements for the algorithm
size_t KMeans::estimateMemoryRequirements() {
    size_t memoryNeeded = 0;
    
    // Points in SoA format
    memoryNeeded += numPoints * dimensions * sizeof(float);
    
    // Centroids in SoA format
    memoryNeeded += dimensions * k * sizeof(float);
    
    // Centroid distances (if using triangle inequality)
    if (useTriangleInequality) {
        memoryNeeded += k * k * sizeof(float);
    }
    
    // Assignments
    memoryNeeded += numPoints * sizeof(int);
    
    // Point-centroid distances (if using triangle inequality)
    if (useTriangleInequality) {
        memoryNeeded += numPoints * k * sizeof(float);
    }
    
    // Other buffers (counts, changes)
    memoryNeeded += k * sizeof(int);
    memoryNeeded += sizeof(int);
    
    // Temporary memory for reduction operations (block sums, counts)
    int threadsPerBlock = 256;
    int numBlocks = (numPoints + threadsPerBlock - 1) / threadsPerBlock;
    memoryNeeded += numBlocks * dimensions * k * sizeof(float); // Block sums
    memoryNeeded += numBlocks * k * sizeof(int); // Block counts
    
    // Add some safety margin (20%)
    memoryNeeded = static_cast<size_t>(memoryNeeded * 1.2);
    
    return memoryNeeded;
}

// Get available GPU memory
size_t KMeans::getAvailableGPUMemory() {
    size_t free, total;
    ::getAvailableGPUMemory(&free, &total);
    return free;
}

// Determine batch size based on available memory
void KMeans::determineBatchSize() {
    if (batchSize > 0) {
        // Manual batch size override
        std::cout << "Using manually specified batch size: " << batchSize << std::endl;
        return;
    }
    
    if (!useGPU) {
        // No batching needed for CPU
        batchSize = 0;
        return;
    }
    
    size_t availableMemory = getAvailableGPUMemory();
    size_t totalMemoryNeeded = estimateMemoryRequirements();
    
    std::cout << "Memory estimate: " << totalMemoryNeeded / (1024 * 1024) 
              << " MB needed, " << availableMemory / (1024 * 1024) 
              << " MB available" << std::endl;
    
    if (totalMemoryNeeded <= availableMemory) {
        // We can process the entire dataset at once
        std::cout << "Sufficient GPU memory available. Processing entire dataset at once." << std::endl;
        batchSize = 0; // 0 means use entire dataset
    } else {
        // Calculate how many points we can process at once
        // Base calculation on points' memory usage
        size_t pointMemory = dimensions * sizeof(float);
        size_t additionalPerPointMemory = sizeof(int);
        if (useTriangleInequality) {
            additionalPerPointMemory += k * sizeof(float);
        }
        size_t memoryPerPoint = pointMemory + additionalPerPointMemory;
        
        // Reserve memory for non-point data (centroids, etc.)
        size_t fixedMemory = dimensions * k * sizeof(float) * 2; // Include space for accumulated centroids
        fixedMemory += k * sizeof(int) * 2; // Include space for accumulated counts
        if (useTriangleInequality) {
            fixedMemory += k * k * sizeof(float);
        }
        fixedMemory += sizeof(int) * 2; // Changes counter and other misc
        
        // Temporary memory for reduction
        int threadsPerBlock = 256;
        int estimatedNumBlocks = 1000; // Estimate for a reasonable batch size
        fixedMemory += estimatedNumBlocks * dimensions * k * sizeof(float); // Block sums
        fixedMemory += estimatedNumBlocks * k * sizeof(int); // Block counts
        
        size_t memoryForPoints = availableMemory * 0.8 - fixedMemory; // Use 80% of available memory
        int calculatedBatchSize = static_cast<int>(memoryForPoints / memoryPerPoint);
        
        // Ensure batch size is reasonable (at least 1000 points, at most the entire dataset)
        batchSize = std::max(1000, std::min(calculatedBatchSize, static_cast<int>(numPoints)));
        
        std::cout << "Using mini-batch processing with batch size: " << batchSize << std::endl;
    }
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

// Allocate GPU memory - Updated for SoA layout and mini-batching
void KMeans::allocateGPUMemory() {
    if (!useGPU) return;
    
    numPoints = points.size();
    dimensions = points[0].features.size();
    
    // Determine batch size based on available memory
    determineBatchSize();
    
    // Allocate memory for centroids in SoA format
    cudaMalloc(&d_centroids, dimensions * k * sizeof(float));
    
    // If using mini-batching, allocate memory for accumulation
    if (batchSize > 0) {
        cudaMalloc(&d_accumulated_centroids, dimensions * k * sizeof(float));
        cudaMalloc(&d_accumulated_counts, k * sizeof(int));
        
        // Initialize accumulated centroids and counts to zero
        cudaMemset(d_accumulated_centroids, 0, dimensions * k * sizeof(float));
        cudaMemset(d_accumulated_counts, 0, k * sizeof(int));
    }
    
    // Allocate memory for centroid distances (if using triangle inequality)
    if (useTriangleInequality) {
        cudaMalloc(&d_centroidDistances, k * k * sizeof(float));
    }
    
    // Always allocate memory for all assignments
    cudaMalloc(&d_assignments, numPoints * sizeof(int));
    
    // Allocate memory for changes counter
    cudaMalloc(&d_changes, sizeof(int));
    
    // Allocate memory for cluster counts
    cudaMalloc(&d_counts, k * sizeof(int));
    
    // Initialize assignments to -1
    cudaMemset(d_assignments, -1, numPoints * sizeof(int));
    
    // When using mini-batching, allocate memory for batch processing
    if (batchSize > 0) {
        // Allocate memory only for batch of points in SoA format
        cudaMalloc(&d_points_soa, dimensions * batchSize * sizeof(float));
        
        // Allocate memory for point-centroid distances for the batch
        if (useTriangleInequality) {
            cudaMalloc(&d_pointCentroidDists, batchSize * k * sizeof(float));
        }
    } else {
        // Allocate memory for all points in SoA format
        cudaMalloc(&d_points_soa, dimensions * numPoints * sizeof(float));
        
        // Allocate memory for point-centroid distances for all points
        if (useTriangleInequality) {
            cudaMalloc(&d_pointCentroidDists, numPoints * k * sizeof(float));
        }
    }
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in allocateGPUMemory: " << cudaGetErrorString(error) << std::endl;
    }
}

// Free GPU memory
void KMeans::freeGPUMemory() {
    if (d_points_soa) cudaFree(d_points_soa);
    if (d_centroids) cudaFree(d_centroids);
    if (d_accumulated_centroids) cudaFree(d_accumulated_centroids);
    if (d_accumulated_counts) cudaFree(d_accumulated_counts);
    if (d_centroidDistances) cudaFree(d_centroidDistances);
    if (d_assignments) cudaFree(d_assignments);
    if (d_changes) cudaFree(d_changes);
    if (d_counts) cudaFree(d_counts);
    if (d_pointCentroidDists) cudaFree(d_pointCentroidDists);
    
    d_points_soa = nullptr;
    d_centroids = nullptr;
    d_accumulated_centroids = nullptr;
    d_accumulated_counts = nullptr;
    d_centroidDistances = nullptr;
    d_assignments = nullptr;
    d_changes = nullptr;
    d_counts = nullptr;
    d_pointCentroidDists = nullptr;
}

// Copy initial data to GPU - Updated for SoA layout
void KMeans::copyInitialDataToGPU() {
    if (!useGPU) return;
    
    // Debug info
    std::cout << "Converting data to SoA format for GPU - Points: " << numPoints 
              << ", Dimensions: " << dimensions << ", Clusters: " << k << std::endl;
    
    // Copy centroids to device in SoA format
    std::vector<float> h_centroids_flat(dimensions * k);
    for (size_t d = 0; d < dimensions; d++) {
        for (int c = 0; c < k; c++) {
            h_centroids_flat[d * k + c] = static_cast<float>(centroids[c][d]);
        }
    }
    cudaMemcpy(d_centroids, h_centroids_flat.data(), dimensions * k * sizeof(float), cudaMemcpyHostToDevice);
    
    // If not using mini-batching, copy all points to device now
    if (batchSize == 0) {
        std::vector<float> h_points_soa(dimensions * numPoints);
        for (size_t d = 0; d < dimensions; d++) {
            for (size_t i = 0; i < numPoints; i++) {
                h_points_soa[d * numPoints + i] = static_cast<float>(points[i].features[d]);
            }
        }
        cudaMemcpy(d_points_soa, h_points_soa.data(), dimensions * numPoints * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    // If using triangle inequality, calculate centroid distances
    if (useTriangleInequality) {
        calculateCentroidDistancesGPU();
    }
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in copyInitialDataToGPU: " << cudaGetErrorString(error) << std::endl;
    }
}

// Copy batch data to GPU for mini-batching
void KMeans::copyBatchDataToGPU(int startIdx, int currentBatchSize) {
    if (!useGPU || batchSize == 0) return;
    
    // Create and copy points for this batch
    std::vector<float> h_batch_points_soa(dimensions * currentBatchSize);
    
    for (size_t d = 0; d < dimensions; d++) {
        for (int i = 0; i < currentBatchSize; i++) {
            int globalIdx = startIdx + i;
            if (globalIdx < numPoints) {
                h_batch_points_soa[d * currentBatchSize + i] = static_cast<float>(points[globalIdx].features[d]);
            }
        }
    }
    
    cudaMemcpy(d_points_soa, h_batch_points_soa.data(), dimensions * currentBatchSize * sizeof(float), cudaMemcpyHostToDevice);
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in copyBatchDataToGPU: " << cudaGetErrorString(error) << std::endl;
    }
}

// Helper method to verify data transfer to GPU
void KMeans::verifyDataTransfer() {
    // Check a few random points to make sure they transferred correctly
    std::vector<float> test_points(dimensions);
    std::vector<float> test_centroids(k);
    
    // Check first dimension of points (only if we're not using mini-batching)
    if (batchSize == 0) {
        cudaMemcpy(test_points.data(), d_points_soa, dimensions * sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << "Verifying points data transfer (first dimension):" << std::endl;
        for (int i = 0; i < std::min(5, static_cast<int>(numPoints)); i++) {
            float device_value = test_points[i];
            float host_value = static_cast<float>(points[i].features[0]);
            std::cout << "Point " << i << " dimension 0: Host=" << host_value << ", Device=" << device_value << std::endl;
        }
    }
    
    // Check first dimension of centroids
    cudaMemcpy(test_centroids.data(), d_centroids, k * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Verifying centroids data transfer (first dimension):" << std::endl;
    for (int i = 0; i < std::min(5, k); i++) {
        float device_value = test_centroids[i];
        float host_value = static_cast<float>(centroids[i][0]);
        std::cout << "Centroid " << i << " dimension 0: Host=" << host_value << ", Device=" << device_value << std::endl;
    }
}

// Copy final results from GPU - Updated for SoA layout
void KMeans::copyFinalResultsFromGPU() {
    if (!useGPU) return;
    
    // Copy assignments back to host
    std::vector<int> h_assignments(numPoints);
    cudaMemcpy(h_assignments.data(), d_assignments, numPoints * sizeof(int), cudaMemcpyDeviceToHost);
    
    for (size_t i = 0; i < numPoints; i++) {
        points[i].cluster = h_assignments[i];
    }
    
    // Copy centroids back to host
    std::vector<float> h_centroids_flat(dimensions * k);
    cudaMemcpy(h_centroids_flat.data(), d_centroids, dimensions * k * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int c = 0; c < k; c++) {
        for (size_t d = 0; d < dimensions; d++) {
            centroids[c][d] = static_cast<double>(h_centroids_flat[d * k + c]);
        }
    }
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in copyFinalResultsFromGPU: " << cudaGetErrorString(error) << std::endl;
    }
}

// Assign clusters - CPU version
int KMeans::assignClustersCPU() {
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
        
        // Store distance to assigned centroid for triangle inequality optimization
        point.distToAssignedCentroid = minDist;
    }
    
    return changes;
}

// Assign clusters - CPU version with triangle inequality
int KMeans::assignClustersCPUWithTriangleInequality() {
    int changes = 0;
    
    // First, update distances between centroids
    calculateCentroidDistances();
    
    for (auto& point : points) {
        double minDist = std::numeric_limits<double>::max();
        int closestCluster = -1;
        int currentCluster = point.cluster;
        
        // If point is already assigned to a cluster, calculate distance to current centroid
        if (currentCluster >= 0) {
            double currentDist = distance(point.features, centroids[currentCluster]);
            point.distToAssignedCentroid = currentDist;
            minDist = currentDist;
            closestCluster = currentCluster;
        }
        
        for (int i = 0; i < k; i++) {
            // Skip the current cluster as we already calculated its distance
            if (i == currentCluster) {
                continue;
            }
            
            // Use triangle inequality to avoid unnecessary distance calculations
            // If d(centroid_i, centroid_current) > 2 * d(point, centroid_current),
            // then centroid_i cannot be closer to the point than centroid_current
            if (currentCluster >= 0 && 
                centroidDistances[i][currentCluster] >= 2.0 * point.distToAssignedCentroid) {
                continue;
            }
            
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
            point.distToAssignedCentroid = minDist;
            changes++;
        }
    }
    
    return changes;
}

// Assign clusters - GPU version - Updated for SoA layout
int KMeans::assignClustersGPU() {
    // If we're using mini-batching, this should never be called directly
    if (batchSize > 0) {
        std::cerr << "Error: assignClustersGPU called directly with mini-batching enabled" << std::endl;
        return 0;
    }
    
    // Call CUDA kernel - all processing stays on GPU
    int changes = assignClustersKernel(
        d_points_soa, d_centroids, d_assignments, d_changes,
        static_cast<int>(numPoints), k, static_cast<int>(dimensions)
    );
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in assignClustersGPU: " << cudaGetErrorString(error) << std::endl;
    }
    
    return changes;
}

// Assign clusters - GPU version with triangle inequality - Updated for SoA layout
int KMeans::assignClustersGPUWithTriangleInequality() {
    // If we're using mini-batching, this should never be called directly
    if (batchSize > 0) {
        std::cerr << "Error: assignClustersGPUWithTriangleInequality called directly with mini-batching enabled" << std::endl;
        return 0;
    }
    
    // Update centroid distances
    calculateCentroidDistancesGPU();
    
    // Call CUDA kernel with triangle inequality optimization
    int changes = assignClustersWithTriangleInequalityKernel(
        d_points_soa, d_centroids, d_centroidDistances, d_pointCentroidDists, 
        d_assignments, d_changes, static_cast<int>(numPoints), k, 
        static_cast<int>(dimensions)
    );
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in assignClustersGPUWithTriangleInequality: " << cudaGetErrorString(error) << std::endl;
    }
    
    return changes;
}

// Assign a batch of points to clusters - GPU version
int KMeans::assignClustersBatchGPU(int startIdx, int currentBatchSize) {
    // Call CUDA kernel for the batch
    int changes = assignClustersKernel(
        d_points_soa, d_centroids, d_assignments, d_changes,
        currentBatchSize, k, static_cast<int>(dimensions), startIdx
    );
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in assignClustersBatchGPU: " << cudaGetErrorString(error) << std::endl;
    }
    
    return changes;
}

// Assign a batch of points to clusters - GPU version with triangle inequality
int KMeans::assignClustersBatchGPUWithTriangleInequality(int startIdx, int currentBatchSize) {
    // Update centroid distances if needed
    calculateCentroidDistancesGPU();
    
    // Call CUDA kernel for the batch with triangle inequality
    int changes = assignClustersWithTriangleInequalityKernel(
        d_points_soa, d_centroids, d_centroidDistances, d_pointCentroidDists,
        d_assignments, d_changes, currentBatchSize, k, 
        static_cast<int>(dimensions), startIdx
    );
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in assignClustersBatchGPUWithTriangleInequality: " << cudaGetErrorString(error) << std::endl;
    }
    
    return changes;
}

// Assign clusters - dispatcher
int KMeans::assignClusters() {
    if (useGPU) {
        if (batchSize > 0) {
            // When using mini-batching, we don't assign all clusters at once
            // This is handled by runAlgorithmWithMiniBatching
            std::cerr << "Error: assignClusters called directly with mini-batching enabled" << std::endl;
            return 0;
        } else {
            if (useTriangleInequality) {
                return assignClustersGPUWithTriangleInequality();
            } else {
                return assignClustersGPU();
            }
        }
    } else {
        if (useTriangleInequality) {
            return assignClustersCPUWithTriangleInequality();
        } else {
            return assignClustersCPU();
        }
    }
}

// Assign a batch of points to clusters - dispatcher
int KMeans::assignClustersBatch(int startIdx, int currentBatchSize) {
    if (useGPU) {
        if (useTriangleInequality) {
            return assignClustersBatchGPUWithTriangleInequality(startIdx, currentBatchSize);
        } else {
            return assignClustersBatchGPU(startIdx, currentBatchSize);
        }
    } else {
        // CPU version - not implemented as it doesn't need batching
        std::cerr << "Error: CPU version does not support batch processing" << std::endl;
        return 0;
    }
}

// Update centroids - CPU version
void KMeans::updateCentroidsCPU() {
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
    
    // If using triangle inequality, update centroid distances
    if (useTriangleInequality) {
        calculateCentroidDistances();
    }
}

// Update centroids - GPU version - Updated for SoA layout and using two-step reduction
void KMeans::updateCentroidsGPU() {
    // If we're using mini-batching, this should never be called directly
    if (batchSize > 0) {
        std::cerr << "Error: updateCentroidsGPU called directly with mini-batching enabled" << std::endl;
        return;
    }
    
    // Call CUDA kernel with two-step reduction
    updateCentroidsKernel(
        d_points_soa, d_centroids, d_assignments, d_counts,
        static_cast<int>(numPoints), k, static_cast<int>(dimensions)
    );
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in updateCentroidsGPU: " << cudaGetErrorString(error) << std::endl;
    }
    
    // If using triangle inequality, update centroid distances
    if (useTriangleInequality) {
        calculateCentroidDistancesGPU();
    }
}

// Update centroids for a batch - GPU version
void KMeans::updateCentroidsBatchGPU(int startIdx, int currentBatchSize) {
    // We need a temporary buffer for batch centroids
    float* d_batch_centroids = nullptr;
    int* d_batch_counts = nullptr;
    
    // Allocate memory for batch centroids and counts
    cudaMalloc(&d_batch_centroids, dimensions * k * sizeof(float));
    cudaMalloc(&d_batch_counts, k * sizeof(int));
    
    // Call CUDA kernel to calculate batch centroids
    updateCentroidsKernel(
        d_points_soa, d_batch_centroids, d_assignments, d_batch_counts,
        currentBatchSize, k, static_cast<int>(dimensions), startIdx
    );
    
    // Accumulate batch centroids into accumulated centroids
    accumulateBatchCentroidsKernel(
        d_batch_centroids, d_batch_counts, d_accumulated_centroids, d_accumulated_counts,
        k, static_cast<int>(dimensions)
    );
    
    // Free temporary memory
    cudaFree(d_batch_centroids);
    cudaFree(d_batch_counts);
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in updateCentroidsBatchGPU: " << cudaGetErrorString(error) << std::endl;
    }
}

// Update centroids - dispatcher
void KMeans::updateCentroids() {
    if (useGPU) {
        if (batchSize > 0) {
            // When using mini-batching, we don't update all centroids at once
            // This is handled by runAlgorithmWithMiniBatching
            std::cerr << "Error: updateCentroids called directly with mini-batching enabled" << std::endl;
        } else {
            updateCentroidsGPU();
        }
    } else {
        updateCentroidsCPU();
    }
}

// Prepare GPU memory (allocate and transfer) - Updated for SoA layout
void KMeans::prepareGPUMemory() {
    if (!useGPU) return;
    
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
    
    std::cout << "Using GPU implementation for k-means clustering with SoA layout" << std::endl;
    std::cout << "Using Two-Step Reduction for centroid updates" << std::endl;
    if (useTriangleInequality) {
        std::cout << "Using Triangle Inequality optimization" << std::endl;
    }
    
    // Allocate GPU memory
    allocateGPUMemory();
    
    // Copy data to GPU
    copyInitialDataToGPU();
}

// Run the k-means algorithm with mini-batching
void KMeans::runAlgorithmWithMiniBatching() {
    if (!useGPU || batchSize <= 0) {
        std::cerr << "Error: Mini-batching requires GPU and batch size > 0" << std::endl;
        return;
    }
    
    int numBatches = (numPoints + batchSize - 1) / batchSize;
    std::cout << "Running K-means with mini-batching (" << numBatches << " batches)" << std::endl;
    
    int iterations = 0;
    int totalChanges;
    
    do {
        totalChanges = 0;
        
        // Reset accumulated centroids and counts
        cudaMemset(d_accumulated_centroids, 0, dimensions * k * sizeof(float));
        cudaMemset(d_accumulated_counts, 0, k * sizeof(int));
        
        // Process each batch
        for (int b = 0; b < numBatches; b++) {
            int startIdx = b * batchSize;
            int endIdx = std::min((b + 1) * batchSize, static_cast<int>(numPoints));
            int currentBatchSize = endIdx - startIdx;
            
            // Copy current batch to GPU
            copyBatchDataToGPU(startIdx, currentBatchSize);
            
            // Assign clusters for this batch
            int batchChanges = assignClustersBatch(startIdx, currentBatchSize);
            totalChanges += batchChanges;
            
            // Accumulate centroids from this batch
            updateCentroidsBatchGPU(startIdx, currentBatchSize);
        }
        
        // Finalize centroids by dividing by counts
        int maxThreads = 32; // Adjust based on GPU capability
        int blockSizeX = std::min(maxThreads, static_cast<int>(dimensions));
        int blockSizeY = std::min(maxThreads, k);
        int gridSizeX = (dimensions + blockSizeX - 1) / blockSizeX;
        int gridSizeY = (k + blockSizeY - 1) / blockSizeY;
        
        // Call finalization kernel through the wrapper function
        finalizeCentroidsKernel(
            d_centroids, d_accumulated_counts, k, static_cast<int>(dimensions),
            gridSizeX, gridSizeY, blockSizeX, blockSizeY
        );
        
        // If using triangle inequality, update centroid distances
        if (useTriangleInequality) {
            calculateCentroidDistancesGPU();
        }
        
        iterations++;
        
        std::cout << "Iteration " << iterations << ": " << totalChanges << " points changed clusters" << std::endl;
        
    } while (totalChanges > 0 && iterations < maxIterations);
    
    std::cout << "K-means with mini-batching converged after " << iterations << " iterations." << std::endl;
}

// Run the k-means algorithm iterations
void KMeans::runAlgorithm() {
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
    
    // Check if we should use mini-batching
    if (useGPU && batchSize > 0) {
        runAlgorithmWithMiniBatching();
        return;
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
        
        // Check GPU memory for errors (debug only)
        if (useGPU) {
            verifyAssignments(iterations);
        }
        
    } while (changes > 0 && iterations < maxIterations);
    
    std::cout << "K-means converged after " << iterations << " iterations." << std::endl;
}

// Helper method to verify assignments
void KMeans::verifyAssignments(int iteration) {
    if (!useGPU) return;
    
    // Sample a few assignments to check if they make sense
    std::vector<int> sample_assignments(std::min(10, static_cast<int>(numPoints)));
    cudaMemcpy(sample_assignments.data(), d_assignments, sample_assignments.size() * sizeof(int), cudaMemcpyDeviceToHost);
    
    std::cout << "Iteration " << iteration << " sample assignments: ";
    for (int i = 0; i < sample_assignments.size(); i++) {
        std::cout << sample_assignments[i] << " ";
    }
    std::cout << std::endl;
    
    // Check for invalid assignments
    for (int i = 0; i < sample_assignments.size(); i++) {
        if (sample_assignments[i] < 0 || sample_assignments[i] >= k) {
            std::cerr << "ERROR: Invalid assignment detected at index " << i 
                      << ": " << sample_assignments[i] << std::endl;
        }
    }
}

// Retrieve results from GPU
void KMeans::retrieveResultsFromGPU() {
    if (!useGPU) return;
    
    // Copy results back from GPU
    copyFinalResultsFromGPU();
    
    // Free GPU memory
    freeGPUMemory();
}

// Run the k-means algorithm - calls the separate stages
void KMeans::run() {
    // If using GPU, prepare memory
    if (useGPU) {
        prepareGPUMemory();
    } else {
        std::cout << "Using CPU implementation for k-means clustering" << std::endl;
        if (useTriangleInequality) {
            std::cout << "Using Triangle Inequality optimization" << std::endl;
        }
    }
    
    // Run the algorithm
    runAlgorithm();
    
    // If using GPU, retrieve results
    if (useGPU) {
        retrieveResultsFromGPU();
    }
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
        // Skip points with invalid cluster assignments
        if (point.cluster < 0 || point.cluster >= k) {
            continue;
        }
        
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

// Save cluster assignments to CSV file
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

// Public method to run appropriate warmup based on settings
void KMeans::warmupKernels() {
    if (!useGPU) {
        // No warmup needed for CPU
        return;
    }
    
    std::cout << "Warming up GPU kernels..." << std::endl;
    
    // Save current state
    std::vector<int> savedAssignments(numPoints);
    if (batchSize == 0) {
        cudaMemcpy(savedAssignments.data(), d_assignments, numPoints * sizeof(int), cudaMemcpyDeviceToHost);
    }
    
    // Create temporary buffers for warmup to avoid modifying the actual data
    int* temp_assignments = nullptr;
    int* temp_changes = nullptr;
    float* temp_centroids = nullptr;
    
    // Allocate temporary memory
    int warmupSize = std::min(10000, static_cast<int>(numPoints));
    cudaMalloc(&temp_assignments, warmupSize * sizeof(int));
    cudaMalloc(&temp_changes, sizeof(int));
    cudaMalloc(&temp_centroids, dimensions * k * sizeof(float));
    
    // Copy current data to temps
    cudaMemset(temp_assignments, -1, warmupSize * sizeof(int));
    cudaMemcpy(temp_centroids, d_centroids, dimensions * k * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // Call the appropriate warmup functions with temporary buffers
    // Warm up assign clusters kernel
    assignClustersKernel(
        d_points_soa, temp_centroids, temp_assignments, temp_changes,
        warmupSize, k, static_cast<int>(dimensions)
    );
    
    // Warm up update centroids kernel with temporary buffers
    int* temp_counts = nullptr;
    cudaMalloc(&temp_counts, k * sizeof(int));
    
    updateCentroidsKernel(
        d_points_soa, temp_centroids, temp_assignments, temp_counts,
        warmupSize, k, static_cast<int>(dimensions)
    );
    
    // Additional warmup for triangle inequality if enabled
    if (useTriangleInequality) {
        float* temp_centroid_distances = nullptr;
        float* temp_point_centroid_dists = nullptr;
        
        cudaMalloc(&temp_centroid_distances, k * k * sizeof(float));
        cudaMalloc(&temp_point_centroid_dists, warmupSize * k * sizeof(float));
        
        // Warm up centroid distances calculation
        calculateCentroidDistancesKernel(
            temp_centroids, temp_centroid_distances, k, static_cast<int>(dimensions)
        );
        
        // Warm up triangle inequality assignment kernel
        assignClustersWithTriangleInequalityKernel(
            d_points_soa, temp_centroids, temp_centroid_distances, temp_point_centroid_dists,
            temp_assignments, temp_changes, warmupSize, k,
            static_cast<int>(dimensions)
        );
        
        // Free temporary memory
        cudaFree(temp_centroid_distances);
        cudaFree(temp_point_centroid_dists);
    }
    
    // Warm up batch-specific kernels if using mini-batching
    if (batchSize > 0) {
        float* temp_accumulated_centroids = nullptr;
        int* temp_accumulated_counts = nullptr;
        
        cudaMalloc(&temp_accumulated_centroids, dimensions * k * sizeof(float));
        cudaMalloc(&temp_accumulated_counts, k * sizeof(int));
        
        // Warm up batch accumulation
        accumulateBatchCentroidsKernel(
            temp_centroids, temp_counts, temp_accumulated_centroids, temp_accumulated_counts,
            k, static_cast<int>(dimensions)
        );
        
        // Free temporary memory
        cudaFree(temp_accumulated_centroids);
        cudaFree(temp_accumulated_counts);
    }
    
    // Synchronize device to ensure all warmup kernels have completed
    cudaDeviceSynchronize();
    
    // Free temporary memory
    cudaFree(temp_assignments);
    cudaFree(temp_changes);
    cudaFree(temp_centroids);
    cudaFree(temp_counts);
    
    // Restore original assignments - important for correctness!
    if (batchSize == 0) {
        cudaMemcpy(d_assignments, savedAssignments.data(), numPoints * sizeof(int), cudaMemcpyHostToDevice);
    }
    
    // Make sure all operations are complete
    cudaDeviceSynchronize();
    
    // Check for any CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error during warmup: " << cudaGetErrorString(error) << std::endl;
    }
    
    std::cout << "GPU warmup complete." << std::endl;
}
