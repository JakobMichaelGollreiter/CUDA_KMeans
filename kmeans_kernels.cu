#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

// Constant memory for centroids (faster access)
__constant__ float c_centroids[1024]; // Supports up to 1024/dimensions centroids

// CUDA kernel to assign points to nearest centroids
__global__ void assignPointsToClusters(
    const float* points,
    const float* centroids, // Keep this for flexibility, even though we're using constant memory
    int* assignments,
    int* changes,
    int numPoints,
    int numClusters,
    int dimensions
) {
    int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pointIdx >= numPoints) return;
    
    // Get the point
    const float* point = &points[pointIdx * dimensions];
    
    // Find the nearest centroid
    float minDist = FLT_MAX;
    int bestCluster = -1;
    
    // Use shared memory to cache centroids
    extern __shared__ float s_centroids[];
    
    // Collaboratively load centroids into shared memory
    for (int c = threadIdx.x; c < numClusters * dimensions; c += blockDim.x) {
        if (c < numClusters * dimensions) {
            // Use constant memory if available and dimensions aren't too large
            if (numClusters * dimensions <= 1024) {
                s_centroids[c] = c_centroids[c];
            } else {
                s_centroids[c] = centroids[c];
            }
        }
    }
    
    __syncthreads();
    
    for (int c = 0; c < numClusters; c++) {
        const float* centroid = &s_centroids[c * dimensions];
        
        // Calculate the squared distance
        float dist = 0.0f;
        for (int d = 0; d < dimensions; d++) {
            float diff = point[d] - centroid[d];
            dist += diff * diff;
        }
        
        if (dist < minDist) {
            minDist = dist;
            bestCluster = c;
        }
    }
    
    // Check if the assignment changed
    int oldCluster = assignments[pointIdx];
    if (oldCluster != bestCluster) {
        assignments[pointIdx] = bestCluster;
        atomicAdd(changes, 1);
    }
}

// CUDA kernel to reset centroids and counts
__global__ void resetCentroidsAndCounts(
    float* centroids,
    int* counts,
    int numClusters,
    int dimensions
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numClusters * dimensions) return;
    
    centroids[idx] = 0.0f;
    
    // Only reset counts once per cluster
    if (idx < numClusters) {
        counts[idx] = 0;
    }
}

// CUDA kernel to accumulate points into centroids
__global__ void accumulatePointsIntoCentroids(
    const float* points,
    float* centroids,
    const int* assignments,
    int* counts,
    int numPoints,
    int dimensions
) {
    int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pointIdx >= numPoints) return;
    
    int cluster = assignments[pointIdx];
    atomicAdd(&counts[cluster], 1);
    
    // Get the point and its centroid
    const float* point = &points[pointIdx * dimensions];
    float* centroid = &centroids[cluster * dimensions];
    
    // Accumulate the point into its centroid
    for (int d = 0; d < dimensions; d++) {
        atomicAdd(&centroid[d], point[d]);
    }
}

// CUDA kernel to finalize centroids by dividing by counts
__global__ void finalizeCentroids(
    float* centroids,
    const int* counts,
    int numClusters,
    int dimensions
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numClusters * dimensions) return;
    
    int cluster = idx / dimensions;
    int count = counts[cluster];
    
    if (count > 0) {
        centroids[idx] /= count;
    }
    
    // If we're using constant memory, update it
    if (numClusters * dimensions <= 1024) {
        // We can't directly write to constant memory, so this will be done by the host
    }
}

// Host function to assign clusters
extern "C" int assignClustersKernel(
    float* d_points,
    float* d_centroids,
    int* d_assignments,
    int* d_changes,
    int numPoints,
    int numClusters,
    int dimensions
) {
    // Reset the changes counter
    cudaMemset(d_changes, 0, sizeof(int));
    
    // Copy the current centroids to constant memory if they fit
    if (numClusters * dimensions <= 1024) {
        cudaMemcpyToSymbol(c_centroids, d_centroids, numClusters * dimensions * sizeof(float));
    }
    
    // Launch the kernel with shared memory for centroids
    int threadsPerBlock = 256;
    int blocksPerGrid = (numPoints + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemSize = numClusters * dimensions * sizeof(float);
    
    assignPointsToClusters<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
        d_points, d_centroids, d_assignments, d_changes, 
        numPoints, numClusters, dimensions
    );
    
    // Copy the result back (only the change count, not all assignments)
    int changes;
    cudaMemcpy(&changes, d_changes, sizeof(int), cudaMemcpyDeviceToHost);
    
    return changes;
}

// Host function to update centroids
extern "C" void updateCentroidsKernel(
    float* d_points,
    float* d_centroids,
    int* d_assignments,
    int* d_counts,
    int numPoints,
    int numClusters,
    int dimensions
) {
    int threadsPerBlock = 256;
    
    // Reset centroids and counts
    int totalCentroidValues = numClusters * dimensions;
    int resetBlocks = (totalCentroidValues + threadsPerBlock - 1) / threadsPerBlock;
    
    resetCentroidsAndCounts<<<resetBlocks, threadsPerBlock>>>(
        d_centroids, d_counts, numClusters, dimensions
    );
    
    // Accumulate points into centroids
    int accumulateBlocks = (numPoints + threadsPerBlock - 1) / threadsPerBlock;
    
    accumulatePointsIntoCentroids<<<accumulateBlocks, threadsPerBlock>>>(
        d_points, d_centroids, d_assignments, d_counts,
        numPoints, dimensions
    );
    
    // Finalize centroids
    finalizeCentroids<<<resetBlocks, threadsPerBlock>>>(
        d_centroids, d_counts, numClusters, dimensions
    );
    
    // Update constant memory if needed
    if (numClusters * dimensions <= 1024) {
        cudaMemcpyToSymbol(c_centroids, d_centroids, numClusters * dimensions * sizeof(float));
    }
}

// Function to check if CUDA is available
extern "C" bool isCUDAAvailable() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    return (error == cudaSuccess && deviceCount > 0);
}
