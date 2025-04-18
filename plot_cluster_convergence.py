import numpy as np
import matplotlib.pyplot as plt
import subprocess
import re
import os

def run_cpp_kmeans(executable="build/kmeans_serial", seed=42):
    """Run the C++ K-means implementation and extract data points and centroids."""
    if not os.path.exists(executable):
        print(f"Error: {executable} not found. Make sure to build the C++ project first.")
        return None, None, None
    
    try:
        # Run the C++ implementation with the specified seed
        result = subprocess.run([executable, str(seed)], 
                               capture_output=True, text=True, check=True)
        
        # Extract true cluster centers
        true_centers = []
        true_center_pattern = r"Cluster (\d+) true center: \(([-\d\.]+), ([-\d\.]+)\), StdDev: ([-\d\.]+)"
        for match in re.finditer(true_center_pattern, result.stdout):
            x = float(match.group(2))
            y = float(match.group(3))
            true_centers.append([x, y])
        
        # Extract final centroids 
        final_centroids = []
        centroid_pattern = r"Centroid (\d+): \(([-\d\.]+), ([-\d\.]+)\)"
        for match in re.finditer(centroid_pattern, result.stdout):
            x = float(match.group(2))
            y = float(match.group(3))
            final_centroids.append([x, y])
        
        # Extract all data points
        data_points = []
        point_pattern = r"Point \(([-\d\.]+), ([-\d\.]+)\)"
        for match in re.finditer(point_pattern, result.stdout):
            x = float(match.group(1))
            y = float(match.group(2))
            data_points.append([x, y])
        
        return np.array(data_points), np.array(true_centers), np.array(final_centroids)
        
    except Exception as e:
        print(f"Error running C++ implementation: {e}")
        return None, None, None

def plot_kmeans_results(data, true_centers, final_centroids, seed):
    """Plot the data distribution with true and final cluster centers."""
    plt.figure(figsize=(10, 8))
    
    # Plot data points
    plt.scatter(data[:, 0], data[:, 1], alpha=0.7, label='Data Points')
    
    # Plot true centers
    plt.scatter(true_centers[:, 0], true_centers[:, 1], 
               c='green', marker='o', s=200, label='True Centers')
    
    # Plot final centers
    if final_centroids is not None:
        plt.scatter(final_centroids[:, 0], final_centroids[:, 1], 
                   c='red', marker='X', s=200, label='Final Centers')
    
    plt.title(f'K-means++ Clustering Results (seed={seed})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("kmeans_visualization.png")
    plt.close()
    
    print("Visualization saved as 'kmeans_visualization.png'")

def main():
    # Set seed parameter
    seed = 42
    
    # Run C++ implementation to get data points and centroids
    data_points, true_centers, final_centroids = run_cpp_kmeans(seed=seed)
    
    if data_points is None:
        print("Failed to extract data from C++ implementation.")
        return
    
    # Plot results
    plot_kmeans_results(data_points, true_centers, final_centroids, seed)

if __name__ == "__main__":
    main()