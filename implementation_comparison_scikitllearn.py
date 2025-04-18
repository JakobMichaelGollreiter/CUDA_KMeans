#!/usr/bin/env python3
"""
Simple K-means++ Comparison Test

This script compares your C++ K-means++ implementation with scikit-learn's implementation
using the same dataset and parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import subprocess
import re
import os

def generate_data(num_clusters=3, points_per_cluster=10, seed=12345):
    """Generate the same data used in your C++ implementation."""
    # Set the random seed for reproducibility
    np.random.seed(seed)
    
    # Define cluster centers (same as in your C++ example)
    cluster_centers = [
        [2.0, 2.0],     # Cluster 0 center
        [8.0, 8.0],     # Cluster 1 center
        [-5.0, -5.0]    # Cluster 2 center
    ]
    
    # Standard deviations for each cluster
    std_devs = [0.1, 0.2, 0.8]
    
    # Generate data points
    data_points = []
    true_labels = []
    
    # Generate points for each cluster
    for c in range(len(cluster_centers)):
        center = cluster_centers[c]
        std_dev = std_devs[c]
        
        for _ in range(points_per_cluster):
            # Generate point with Gaussian distribution around center
            point = [np.random.normal(center[d], std_dev) for d in range(len(center))]
            data_points.append(point)
            true_labels.append(c)
    
    return np.array(data_points), np.array(true_labels), np.array(cluster_centers)

def run_sklearn_kmeans(data, n_clusters=3, seed=12345):
    """Run scikit-learn's KMeans on the data."""
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=1,  # Use a single initialization for comparison
        random_state=seed,
        max_iter=300,
        tol=1e-4
    )
    kmeans.fit(data)
    
    print("\nScikit-learn K-means++ Results:")
    print("-------------------------------")
    print(f"Number of iterations: {kmeans.n_iter_}")
    print(f"Sum of squared errors: {kmeans.inertia_:.6f}")
    
    # Print centroids
    print("\nEstimated cluster centroids:")
    for i, centroid in enumerate(kmeans.cluster_centers_):
        print(f"Centroid {i}: ({centroid[0]:.6f}, {centroid[1]:.6f})")
    
    # Count points in each cluster
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    print("\nCluster point counts:")
    for i, count in zip(unique, counts):
        print(f"Cluster {i}: {count} points")
    
    return kmeans

def run_cpp_kmeans(executable="build/kmeans_serial", seed=12345):
    """Run the C++ K-means implementation and parse its output."""
    if not os.path.exists(executable):
        print(f"Error: {executable} not found. Make sure to build the C++ project first.")
        return None
    
    try:
        # Run the C++ implementation with the specified seed
        result = subprocess.run([executable, str(seed)], 
                               capture_output=True, text=True, check=True)
        
        # Extract centroids using regex
        centroids = []
        centroid_pattern = r"Centroid (\d+): \(([-\d\.]+), ([-\d\.]+)\)"
        for match in re.finditer(centroid_pattern, result.stdout):
            cluster_id = int(match.group(1))
            x = float(match.group(2))
            y = float(match.group(3))
            centroids.append([x, y])
        
        # Extract SSE
        sse_pattern = r"Sum of Squared Errors: ([-\d\.]+)"
        sse_match = re.search(sse_pattern, result.stdout)
        sse = float(sse_match.group(1)) if sse_match else None
        
        # Extract iteration count (if available)
        iter_pattern = r"K-means converged after (\d+) iterations"
        iter_match = re.search(iter_pattern, result.stdout)
        iterations = int(iter_match.group(1)) if iter_match else None
        
        # Extract cluster counts
        cluster_counts = []
        count_pattern = r"Cluster (\d+): (\d+) points"
        for match in re.finditer(count_pattern, result.stdout):
            cluster_id = int(match.group(1))
            count = int(match.group(2))
            cluster_counts.append((cluster_id, count))
        
        # Sort by cluster ID
        cluster_counts.sort()
        
        print("\nC++ K-means++ Results:")
        print("-----------------------")
        if iterations is not None:
            print(f"Number of iterations: {iterations}")
        if sse is not None:
            print(f"Sum of squared errors: {sse:.6f}")
        
        # Print centroids
        print("\nEstimated cluster centroids:")
        for i, centroid in enumerate(centroids):
            print(f"Centroid {i}: ({centroid[0]:.6f}, {centroid[1]:.6f})")
        
        # Print cluster counts
        print("\nCluster point counts:")
        for cluster_id, count in cluster_counts:
            print(f"Cluster {cluster_id}: {count} points")
        
        return {
            'centroids': np.array(centroids),
            'sse': sse,
            'iterations': iterations,
            'cluster_counts': cluster_counts
        }
        
    except Exception as e:
        print(f"Error running C++ implementation: {e}")
        return None

def compare_results(sklearn_kmeans, cpp_results):
    """Compare the results from both implementations."""
    if cpp_results is None:
        print("Cannot compare results: C++ implementation failed to run.")
        return
    
    # Get scikit-learn centroids
    sklearn_centroids = sklearn_kmeans.cluster_centers_
    cpp_centroids = cpp_results['centroids']
    
    # Calculate distances between all pairs of centroids
    distances = np.zeros((len(cpp_centroids), len(sklearn_centroids)))
    
    for i, c1 in enumerate(cpp_centroids):
        for j, c2 in enumerate(sklearn_centroids):
            distances[i, j] = np.sqrt(np.sum((c1 - c2) ** 2))
    
    # Find the best matching of centroids
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(distances)
    
    # Calculate metrics
    total_distance = distances[row_ind, col_ind].sum()
    max_distance = distances[row_ind, col_ind].max()
    avg_distance = total_distance / len(row_ind)
    
    print("\nCentroid Comparison:")
    print("--------------------")
    for i, (cpp_idx, sklearn_idx) in enumerate(zip(row_ind, col_ind)):
        cpp_centroid = cpp_centroids[cpp_idx]
        sklearn_centroid = sklearn_centroids[sklearn_idx]
        dist = distances[cpp_idx, sklearn_idx]
        
        print(f"Match {i+1}:")
        print(f"  C++ Centroid {cpp_idx}: ({cpp_centroid[0]:.6f}, {cpp_centroid[1]:.6f})")
        print(f"  Sklearn Centroid {sklearn_idx}: ({sklearn_centroid[0]:.6f}, {sklearn_centroid[1]:.6f})")
        print(f"  Distance: {dist:.6f}")
    
    print(f"\nTotal distance: {total_distance:.6f}")
    print(f"Maximum distance: {max_distance:.6f}")
    print(f"Average distance: {avg_distance:.6f}")
    
    # Compare SSE
    sklearn_sse = sklearn_kmeans.inertia_
    cpp_sse = cpp_results['sse']
    sse_diff = abs(sklearn_sse - cpp_sse) if cpp_sse is not None else None
    sse_pct_diff = (sse_diff / sklearn_sse) * 100 if cpp_sse is not None else None
    
    print("\nSum of Squared Errors (SSE) Comparison:")
    print("----------------------------------------")
    print(f"Scikit-learn SSE: {sklearn_sse:.6f}")
    if cpp_sse is not None:
        print(f"C++ SSE: {cpp_sse:.6f}")
        print(f"Absolute difference: {sse_diff:.6f}")
        print(f"Percentage difference: {sse_pct_diff:.4f}%")
    
    # Determine if implementations are equivalent
    if max_distance < 0.1 and (sse_pct_diff is None or sse_pct_diff < 1.0):
        print("\n✅ Conclusion: The implementations appear to be equivalent!")
        print("   (Centroids are within 0.1 distance and SSE differs by less than 1%)")
    else:
        print("\n⚠️ Conclusion: There are some differences between the implementations.")
        print("   This could be due to:")
        print("   - Different initialization strategies")
        print("   - Different convergence criteria")
        print("   - Numerical precision differences")
        print("   - Random seed handling differences")
    
    return {
        'max_distance': max_distance,
        'avg_distance': avg_distance,
        'sse_diff_pct': sse_pct_diff
    }

def visualize_comparison(data, sklearn_kmeans, cpp_results, true_centers):
    """Create a visual comparison of both implementations."""
    if cpp_results is None:
        print("Cannot visualize: C++ implementation failed to run.")
        return
    
    plt.figure(figsize=(15, 5))
    
    # Plot original data with true centers
    plt.subplot(1, 3, 1)
    plt.scatter(data[:, 0], data[:, 1], c=sklearn_kmeans.labels_, cmap='viridis', alpha=0.7)
    plt.scatter(true_centers[:, 0], true_centers[:, 1], c='red', marker='X', s=100)
    plt.title("Original Data\nwith True Cluster Centers")
    plt.xlabel("X")
    plt.ylabel("Y")
    
    # Plot scikit-learn results
    plt.subplot(1, 3, 2)
    plt.scatter(data[:, 0], data[:, 1], c=sklearn_kmeans.labels_, cmap='viridis', alpha=0.7)
    plt.scatter(sklearn_kmeans.cluster_centers_[:, 0], sklearn_kmeans.cluster_centers_[:, 1], 
               c='red', marker='X', s=100)
    plt.title("scikit-learn K-means++")
    plt.xlabel("X")
    plt.ylabel("Y")
    
    # Plot C++ results (approximate visualization)
    plt.subplot(1, 3, 3)
    plt.scatter(data[:, 0], data[:, 1], c=sklearn_kmeans.labels_, cmap='viridis', alpha=0.7)
    plt.scatter(cpp_results['centroids'][:, 0], cpp_results['centroids'][:, 1], 
               c='red', marker='X', s=100)
    plt.title("C++ K-means++")
    plt.xlabel("X")
    plt.ylabel("Y")
    
    plt.tight_layout()
    plt.savefig("kmeans_comparison.png")
    plt.close()
    
    print("\nVisualization saved as 'kmeans_comparison.png'")

def main():
    """Main function to run the comparison test."""
    print("=" * 60)
    print("K-means++ Implementation Comparison Test".center(60))
    print("=" * 60)
    
    # Set parameters
    num_clusters = 3
    points_per_cluster = 2
    seed = 42
    
    # Generate data
    print(f"Generating synthetic data with {num_clusters} clusters, "
          f"{points_per_cluster} points per cluster, seed={seed}...")
    data, true_labels, true_centers = generate_data(
        num_clusters=num_clusters, 
        points_per_cluster=points_per_cluster,
        seed=seed
    )
    print(f"Generated {len(data)} data points.")
    
    # Run scikit-learn K-means++
    sklearn_kmeans = run_sklearn_kmeans(data, n_clusters=num_clusters, seed=seed)
    
    # Run C++ K-means++
    cpp_results = run_cpp_kmeans(seed=seed)
    
    # Compare results
    compare_results(sklearn_kmeans, cpp_results)
    
    # Visualize comparison
    visualize_comparison(data, sklearn_kmeans, cpp_results, true_centers)

if __name__ == "__main__":
    main()