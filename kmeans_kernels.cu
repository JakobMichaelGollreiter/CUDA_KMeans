#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <float.h>
#include <math.h>

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
        float dist_squared = 0.0f;
        for (int d = 0; d < dimensions; d++) {
            // SoA layout for both points and centroids
            // For point access: points_soa[d * numPoints + pointIdx]
            // For centroid access: s_centroids[d * numClusters + c]
            float diff = points_soa[d * numPoints + pointIdx] - s_centroids[d * numClusters + c];
            dist_squared += diff * diff;
        }
        
        // Convert to regular Euclidean distance
        float dist = sqrtf(dist_squared);
        
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
    float dist_squared = 0.0f;
    for (int d = 0; d < dimensions; d++) {
        // SoA layout for centroids: centroids[d * numClusters + c]
        float diff = centroids[d * numClusters + i] - centroids[d * numClusters + j];
        dist_squared += diff * diff;
    }
    
    // Convert to regular Euclidean distance
    centroidDistances[i * numClusters + j] = sqrtf(dist_squared);
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
        float dist_squared = 0.0f;
        for (int d = 0; d < dimensions; d++) {
            // SoA layout for both points and centroids
            float diff = points_soa[d * numPoints + pointIdx] - s_centroids[d * numClusters + currentCluster];
            dist_squared += diff * diff;
        }
        
        // Convert to regular Euclidean distance
        currentDist = sqrtf(dist_squared);
        
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
            
            // If d(centroid_c, centroid_current) > 2 * d(point, centroid_current),
            // then centroid_c cannot be closer to the point than centroid_current
            if (centroidDist > 2.0f * currentDist) {
                continue;
            }
        }
        
        // Calculate distance to this centroid
        float dist_squared = 0.0f;
        for (int d = 0; d < dimensions; d++) {
            // SoA layout for both points and centroids
            float diff = points_soa[d * numPoints + pointIdx] - s_centroids[d * numClusters + c];
            dist_squared += diff * diff;
        }
        
        // Convert to regular Euclidean distance
        float dist = sqrtf(dist_squared);
        
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

// First step: Calculate local sums per block
__global__ void accumulatePointsLocalSums(
    const float* points_soa,  // Points in SoA format
    const int* assignments,
    float* blockSums,         // Block-level sums [blockIdx * numClusters * dimensions]
    int* blockCounts,         // Block-level counts [blockIdx * numClusters]
    int numPoints,
    int numClusters,
    int dimensions
) {
    // Get global thread ID
    int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for local sums and counts within the block
    extern __shared__ float s_data[];
    float* s_sums = s_data;                                     // Size: dimensions * numClusters
    int* s_counts = (int*)&s_data[dimensions * numClusters];    // Size: numClusters
    
    // Initialize shared memory
    for (int i = threadIdx.x; i < dimensions * numClusters; i += blockDim.x) {
        s_sums[i] = 0.0f;
    }
    
    for (int i = threadIdx.x; i < numClusters; i += blockDim.x) {
        s_counts[i] = 0;
    }
    
    __syncthreads();
    
    // Accumulate points only if within range
    if (pointIdx < numPoints) {
        int cluster = assignments[pointIdx];
        if (cluster >= 0 && cluster < numClusters) {
            // Increment cluster count for this block
            atomicAdd(&s_counts[cluster], 1);
            
            // Accumulate point features (still uses atomics but within shared memory - much faster)
            for (int d = 0; d < dimensions; d++) {
                atomicAdd(&s_sums[d * numClusters + cluster], points_soa[d * numPoints + pointIdx]);
            }
        }
    }
    
    __syncthreads();
    
    // Write block results to global memory
    // Each block writes its own section of the blockSums and blockCounts arrays
    for (int i = threadIdx.x; i < dimensions * numClusters; i += blockDim.x) {
        blockSums[blockIdx.x * dimensions * numClusters + i] = s_sums[i];
    }
    
    for (int i = threadIdx.x; i < numClusters; i += blockDim.x) {
        blockCounts[blockIdx.x * numClusters + i] = s_counts[i];
    }
}

// Second step: Merge block-level results into final centroids
__global__ void mergeBlockSums(
    float* blockSums,     // Block-level sums
    int* blockCounts,     // Block-level counts
    float* centroids,     // Final centroids
    int* counts,          // Final counts
    int numBlocks,
    int numClusters,
    int dimensions
) {
    // Each thread handles one centroid dimension
    int dim = blockIdx.x;
    int cluster = threadIdx.x;
    
    if (dim >= dimensions || cluster >= numClusters) return;
    
    float sum = 0.0f;
    
    // Accumulate from all blocks
    for (int b = 0; b < numBlocks; b++) {
        sum += blockSums[b * dimensions * numClusters + dim * numClusters + cluster];
    }
    
    // Store accumulated sum
    centroids[dim * numClusters + cluster] = sum;
    
    // For the first dimension, also accumulate counts
    if (dim == 0) {
        int count = 0;
        for (int b = 0; b < numBlocks; b++) {
            count += blockCounts[b * numClusters + cluster];
        }
        counts[cluster] = count;
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
    // int threadsPerBlock = 256;
    int threadsPerBlock = 1024;
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

// Host function to update centroids - MODIFIED to use two-step reduction
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
    
    // First step: Calculate local sums per block
    int accumulateThreads = 256;
    int accumulateBlocks = (numPoints + accumulateThreads - 1) / accumulateThreads;
    
    // Allocate memory for block-level results
    float* d_blockSums;
    int* d_blockCounts;
    cudaMalloc(&d_blockSums, accumulateBlocks * dimensions * numClusters * sizeof(float));
    cudaMalloc(&d_blockCounts, accumulateBlocks * numClusters * sizeof(int));
    
    // Calculate shared memory size
    int sharedMemSize = (dimensions * numClusters * sizeof(float)) + (numClusters * sizeof(int));
    
    // Launch kernel to calculate local sums
    accumulatePointsLocalSums<<<accumulateBlocks, accumulateThreads, sharedMemSize>>>(
        d_points_soa, d_assignments, d_blockSums, d_blockCounts,
        numPoints, numClusters, dimensions
    );
    
    // Check for errors
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error in accumulatePointsLocalSums: %s\n", cudaGetErrorString(error));
    }
    
    // Second step: Merge block results
    dim3 mergeBlock(numClusters);
    dim3 mergeGrid(dimensions);
    
    mergeBlockSums<<<mergeGrid, mergeBlock>>>(
        d_blockSums, d_blockCounts, d_centroids, d_counts,
        accumulateBlocks, numClusters, dimensions
    );
    
    // Check for errors
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error in mergeBlockSums: %s\n", cudaGetErrorString(error));
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
    
    // Free temporary memory
    cudaFree(d_blockSums);
    cudaFree(d_blockCounts);
    
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
