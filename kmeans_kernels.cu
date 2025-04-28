#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <float.h>

// Constant memory for centroids (faster access)
__constant__ float c_centroids[1024]; // Supports up to 1024/dimensions centroids
__constant__ float c_centroidDistances[1024]; // Supports up to 32x32 centroids

// CUDA kernel to assign points to nearest centroids
__global__ void assignPointsToClusters(
    const float* points_soa,  // Points in SoA format: dimension-major ordering
    const float* centroids,   // Centroids in SoA format
    int* assignments,
    int* changes,
    int numPoints,
    int numClusters,
    int dimensions
) {
    int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pointIdx >= numPoints) return;
    
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
        // Calculate the squared distance
        float dist = 0.0f;
        for (int d = 0; d < dimensions; d++) {
            // SoA layout for both points and centroids
            // For point access: points_soa[d * numPoints + pointIdx]
            // For centroid access: s_centroids[d * numClusters + c]
            float diff = points_soa[d * numPoints + pointIdx] - s_centroids[d * numClusters + c];
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

// CUDA kernel to calculate distances between centroids
__global__ void calculateCentroidDistances(
    const float* centroids,  // Centroids in SoA format
    float* centroidDistances,
    int numClusters,
    int dimensions
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= numClusters || j >= numClusters) return;
    
    // Skip diagonal (distance to self is 0)
    if (i == j) {
        centroidDistances[i * numClusters + j] = 0.0f;
        return;
    }
    
    // Calculate squared distance between centroids
    float dist = 0.0f;
    for (int d = 0; d < dimensions; d++) {
        // SoA layout for centroids: centroids[d * numClusters + c]
        float diff = centroids[d * numClusters + i] - centroids[d * numClusters + j];
        dist += diff * diff;
    }
    
    // Store squared distance (no need for sqrt as we compare squared distances)
    centroidDistances[i * numClusters + j] = dist;
}

// CUDA kernel to assign points to nearest centroids using triangle inequality
__global__ void assignPointsToClustersWithTriangleInequality(
    const float* points_soa,  // Points in SoA format
    const float* centroids,   // Centroids in SoA format
    const float* centroidDistances,
    float* pointCentroidDists,  // Store distances between points and centroids
    int* assignments,
    int* changes,
    int numPoints,
    int numClusters,
    int dimensions
) {
    int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pointIdx >= numPoints) return;
    
    // Get current assignment and distance
    int currentCluster = assignments[pointIdx];
    float currentDist = FLT_MAX;
    
    // Use shared memory to cache centroids and centroid distances
    extern __shared__ float s_data[];
    float* s_centroids = s_data;
    float* s_centroidDistances = &s_data[numClusters * dimensions];
    
    // Collaboratively load centroids into shared memory
    for (int c = threadIdx.x; c < numClusters * dimensions; c += blockDim.x) {
        if (c < numClusters * dimensions) {
            if (numClusters * dimensions <= 1024) {
                s_centroids[c] = c_centroids[c];
            } else {
                s_centroids[c] = centroids[c];
            }
        }
    }
    
    // Collaboratively load centroid distances into shared memory
    for (int c = threadIdx.x; c < numClusters * numClusters; c += blockDim.x) {
        if (c < numClusters * numClusters) {
            if (numClusters * numClusters <= 1024) {
                s_centroidDistances[c] = c_centroidDistances[c];
            } else {
                s_centroidDistances[c] = centroidDistances[c];
            }
        }
    }
    
    __syncthreads();
    
    // If the point is already assigned, calculate its distance to current centroid
    if (currentCluster >= 0) {
        currentDist = 0.0f;
        for (int d = 0; d < dimensions; d++) {
            // SoA layout for both points and centroids
            float diff = points_soa[d * numPoints + pointIdx] - s_centroids[d * numClusters + currentCluster];
            currentDist += diff * diff;
        }
        
        // Store the distance
        pointCentroidDists[pointIdx * numClusters + currentCluster] = currentDist;
    }
    
    float minDist = currentDist;
    int bestCluster = currentCluster;
    
    // Check all other centroids
    for (int c = 0; c < numClusters; c++) {
        // Skip current cluster
        if (c == currentCluster) continue;
        
        // If we already have an assignment, use triangle inequality to avoid unnecessary calculations
        if (currentCluster >= 0) {
            float centroidDist = s_centroidDistances[c * numClusters + currentCluster];
            
            // If d(centroid_c, centroid_current) >= 4 * d(point, centroid_current),
            // then centroid_c cannot be closer to the point than centroid_current
            // Note: We use squared distances, so it's 4 * dist instead of 2 * sqrt(dist)
            if (centroidDist >= 4.0f * currentDist) {
                continue;
            }
        }
        
        // Calculate distance to this centroid
        float dist = 0.0f;
        for (int d = 0; d < dimensions; d++) {
            // SoA layout for both points and centroids
            float diff = points_soa[d * numPoints + pointIdx] - s_centroids[d * numClusters + c];
            dist += diff * diff;
        }
        
        // Store the distance
        pointCentroidDists[pointIdx * numClusters + c] = dist;
        
        // Update if closer
        if (dist < minDist) {
            minDist = dist;
            bestCluster = c;
        }
    }
    
    // Check if the assignment changed
    if (currentCluster != bestCluster) {
        assignments[pointIdx] = bestCluster;
        atomicAdd(changes, 1);
    }
}

// CUDA kernel to reset centroids and counts
__global__ void resetCentroidsAndCounts(
    float* centroids,  // Centroids in SoA format
    int* counts,
    int numClusters,
    int dimensions
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dimensions * numClusters) return;
    
    centroids[idx] = 0.0f;
    
    // Only reset counts once per cluster (not for each dimension)
    if (idx < numClusters) {
        counts[idx] = 0;
    }
}

// CUDA kernel to accumulate points into centroids - simplified version with direct atomics
__global__ void accumulatePointsIntoCentroids(
    const float* points_soa,  // Points in SoA format
    float* centroids,        // Centroids in SoA format
    const int* assignments,
    int* counts,
    int numPoints,
    int dimensions,
    int numClusters
) {
    int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pointIdx >= numPoints) return;
    
    int cluster = assignments[pointIdx];
    if (cluster < 0 || cluster >= numClusters) return; // Skip invalid assignments
    
    // Atomically increment the count for this cluster
    atomicAdd(&counts[cluster], 1);
    
    // Accumulate the point's features into the centroid
    for (int d = 0; d < dimensions; d++) {
        atomicAdd(&centroids[d * numClusters + cluster], points_soa[d * numPoints + pointIdx]);
    }
}

// CUDA kernel to finalize centroids by dividing by counts
__global__ void finalizeCentroids(
    float* centroids,  // Centroids in SoA format
    const int* counts,
    int numClusters,
    int dimensions
) {
    int d = blockIdx.x;
    int c = threadIdx.x;
    
    if (d >= dimensions || c >= numClusters) return;
    
    // SoA layout: centroids[d * numClusters + c]
    int count = counts[c];
    if (count > 0) {
        centroids[d * numClusters + c] /= count;
    }
}

// Host function to calculate centroid distances
extern "C" void calculateCentroidDistancesKernel(
    float* d_centroids,
    float* d_centroidDistances,
    int numClusters,
    int dimensions
) {
    dim3 blockSize(16, 16);
    dim3 gridSize((numClusters + blockSize.x - 1) / blockSize.x, 
                  (numClusters + blockSize.y - 1) / blockSize.y);
    
    calculateCentroidDistances<<<gridSize, blockSize>>>(
        d_centroids, d_centroidDistances, numClusters, dimensions
    );
    
    // Copy to constant memory if it fits
    if (numClusters * numClusters <= 1024) {
        cudaMemcpyToSymbol(c_centroidDistances, d_centroidDistances, 
                           numClusters * numClusters * sizeof(float));
    }
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error in calculateCentroidDistancesKernel: %s\n", cudaGetErrorString(error));
    }
}

// Host function to assign clusters using triangle inequality
extern "C" int assignClustersWithTriangleInequalityKernel(
    float* d_points_soa,
    float* d_centroids,
    float* d_centroidDistances,
    float* d_pointCentroidDists,
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
    
    // Launch the kernel with shared memory for centroids and centroid distances
    int threadsPerBlock = 256;
    int blocksPerGrid = (numPoints + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemSize = (numClusters * dimensions + numClusters * numClusters) * sizeof(float);
    
    assignPointsToClustersWithTriangleInequality<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
        d_points_soa, d_centroids, d_centroidDistances, d_pointCentroidDists,
        d_assignments, d_changes, numPoints, numClusters, dimensions
    );
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error in assignClustersWithTriangleInequalityKernel: %s\n", cudaGetErrorString(error));
    }
    
    // Copy the result back (only the change count, not all assignments)
    int changes;
    cudaMemcpy(&changes, d_changes, sizeof(int), cudaMemcpyDeviceToHost);
    
    return changes;
}

// Host function to assign clusters
extern "C" int assignClustersKernel(
    float* d_points_soa,
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
        d_points_soa, d_centroids, d_assignments, d_changes, 
        numPoints, numClusters, dimensions
    );
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error in assignClustersKernel: %s\n", cudaGetErrorString(error));
    }
    
    // Copy the result back (only the change count, not all assignments)
    int changes;
    cudaMemcpy(&changes, d_changes, sizeof(int), cudaMemcpyDeviceToHost);
    
    return changes;
}

// Host function to update centroids
extern "C" void updateCentroidsKernel(
    float* d_points_soa,
    float* d_centroids,
    int* d_assignments,
    int* d_counts,
    int numPoints,
    int numClusters,
    int dimensions
) {
    // Reset centroids and counts
    int threadsPerBlock = 256;
    int totalCentroidValues = numClusters * dimensions;
    int resetBlocks = (totalCentroidValues + threadsPerBlock - 1) / threadsPerBlock;
    
    resetCentroidsAndCounts<<<resetBlocks, threadsPerBlock>>>(
        d_centroids, d_counts, numClusters, dimensions
    );
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error in resetCentroidsAndCounts: %s\n", cudaGetErrorString(error));
    }
    
    // Accumulate points into centroids - using simplified direct atomic version
    int accumulateThreads = 256;
    int accumulateBlocks = (numPoints + accumulateThreads - 1) / accumulateThreads;
    
    accumulatePointsIntoCentroids<<<accumulateBlocks, accumulateThreads>>>(
        d_points_soa, d_centroids, d_assignments, d_counts,
        numPoints, dimensions, numClusters
    );
    
    // Check for errors
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error in accumulatePointsIntoCentroids: %s\n", cudaGetErrorString(error));
    }
    
    // Finalize centroids with one thread per dimension-cluster pair
    dim3 finalizeBlock(numClusters);
    dim3 finalizeGrid(dimensions);
    
    finalizeCentroids<<<finalizeGrid, finalizeBlock>>>(
        d_centroids, d_counts, numClusters, dimensions
    );
    
    // Check for errors
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error in finalizeCentroids: %s\n", cudaGetErrorString(error));
    }
    
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
