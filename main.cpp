#include "kmeans.h"
#include <vector>
#include <iostream>
#include <string>
#include <stdexcept>
#include <sys/stat.h>  // For mkdir
#include <unistd.h>    // For access
#include <chrono>

// Extract filename without extension
std::string getFilenameStem(const std::string& path) {
    size_t lastSlash = path.find_last_of("/\\");
    size_t lastDot = path.find_last_of('.');
    if (lastDot == std::string::npos || lastDot < lastSlash) lastDot = path.length();
    return path.substr(lastSlash + 1, lastDot - lastSlash - 1);
}

// Create directory if it doesn't exist
void makeDirectory(const std::string& dir) {
    if (access(dir.c_str(), F_OK) != 0) {
        mkdir(dir.c_str(), 0755); // 0755 = rwxr-xr-x
    }
}

int main(int argc, char* argv[]) {
    // Check command line arguments
    if (argc < 3 || argc > 7) {
        std::cerr << "Usage: " << argv[0] << " <data_file.csv> <centroids_file.csv> <num_clusters> [max_iterations] [use_gpu] [use_triangle]" << std::endl;
        std::cerr << "  <data_file.csv>     : Path to CSV file containing data points" << std::endl;
        std::cerr << "  <centroids_file.csv>: Path to CSV file containing initial centroids" << std::endl;
        std::cerr << "  <num_clusters>      : Number of clusters (k)" << std::endl;
        std::cerr << "  [max_iterations]    : Maximum iterations (default: 100)" << std::endl;
        std::cerr << "  [use_gpu]           : Use GPU acceleration if available (0 or 1, default: 0)" << std::endl;
        std::cerr << "  [use_triangle]      : Use Triangle Inequality optimization (0 or 1, default: 0)" << std::endl;
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
        
        // Optional argument for max iterations
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
        
        // Optional argument for GPU usage
        bool useGPU = false;  // Default value
        if (argc >= 6) {
            try {
                useGPU = (std::stoi(argv[5]) != 0);
            } catch (const std::exception& e) {
                std::cerr << "Error: Invalid GPU flag. Using default: CPU" << std::endl;
            }
        }
        
        // Optional argument for Triangle Inequality optimization
        bool useTriangleInequality = false;  // Default value
        if (argc >= 7) {
            try {
                useTriangleInequality = (std::stoi(argv[6]) != 0);
            } catch (const std::exception& e) {
                std::cerr << "Error: Invalid Triangle Inequality flag. Using default: No optimization" << std::endl;
            }
        }
        
        // Check if CUDA is available if GPU requested
        if (useGPU && !KMeans::isCUDAAvailable()) {
            std::cerr << "Warning: CUDA is not available on this system. Falling back to CPU implementation." << std::endl;
            useGPU = false;
        }
        
        // Create a KMeans instance
        KMeans kmeans(numClusters, maxIterations, 1e-4, useGPU, useTriangleInequality);
        
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
        
        // Separate data loading and GPU memory preparation from algorithm timing
        if (useGPU) {
            std::cout << "\nPreparing GPU memory (not included in timing)..." << std::endl;
            kmeans.prepareGPUMemory();
            
            // NEW: Warm up GPU kernels before timing
            kmeans.warmupKernels();
        }
        
        // Run the clustering algorithm with timing only the algorithm, not data transfers
        std::cout << "\nRunning K-means clustering..." << std::endl;
        std::cout << (useGPU ? "Using GPU acceleration" : "Using CPU implementation") << std::endl;
        std::cout << (useTriangleInequality ? "Using Triangle Inequality optimization" : "Using standard K-means") << std::endl;

        // Start timing
        auto start = std::chrono::high_resolution_clock::now();

        // Run only the algorithm (Lloyd's iterations)
        kmeans.runAlgorithm();

        // End timing
        auto end = std::chrono::high_resolution_clock::now();

        // If using GPU, copy results back (not included in timing)
        if (useGPU) {
            std::cout << "Copying results from GPU (not included in timing)..." << std::endl;
            kmeans.retrieveResultsFromGPU();
        }

        // Calculate and display the algorithm duration only
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "\nAlgorithm time (excluding transfers): " << elapsed.count() << " seconds" << std::endl;
        std::cout << "Clustering completed." << std::endl;
        std::cout << "------------------------" << std::endl;

        // Get cluster assignments and centroids
        auto assignments = kmeans.getClusterAssignments();
        auto centroids = kmeans.getCentroids();
        
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
        
        // Extract filename
        std::string filename = getFilenameStem(dataFile);

        // Create output directories
        std::string clusteOutputDir = "../label_predictions";
        std::string centroidsOutputDir = "../center_predictions";

        makeDirectory(clusteOutputDir);
        makeDirectory(centroidsOutputDir);

        // Construct file paths
        std::string outputClustersFile = clusteOutputDir + "/" + filename + "_labels.csv";
        std::string outputCentroidsFile = centroidsOutputDir + "/" + filename + "_centers.csv";

        kmeans.saveClusterAssignmentsToCSV(outputClustersFile);
        kmeans.saveCentroidsToCSV(outputCentroidsFile);
            
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
