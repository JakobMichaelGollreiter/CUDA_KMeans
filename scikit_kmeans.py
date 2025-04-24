from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import time
import os
import argparse

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def run_kmeans(dataset_path, init_centers_path, n_clusters, max_iterations=3):
    """Run KMeans clustering and save results."""
    # Extract dataset name from path for naming output files
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    
    # Setup output directories
    centers_dir = 'scikit_center_predictions'
    labels_dir = 'scikit_label_predictions'
    ensure_dir(centers_dir)
    ensure_dir(labels_dir)
    
    # Load data
    df = pd.read_csv(dataset_path, header=0)
    X = df.iloc[:, :-1].values  # Exclude the last column (labels)
    print("Data shape: {}".format(X.shape))
    
    # Load initial centers
    init_df = pd.read_csv(init_centers_path, header=0)
    initial_centers = init_df.values
    print("Initial centers shape: {}".format(initial_centers.shape))
    
    # Time KMeans fitting
    start_time = time.time()
    kmeans = KMeans(n_clusters=n_clusters, init=initial_centers, n_init=1, max_iter=max_iterations)
    kmeans.fit(X)
    elapsed_time = time.time() - start_time
    
    print("KMeans fitting time: {:.6f} seconds".format(elapsed_time))
    print("Number of iterations: {}".format(kmeans.n_iter_))
    
    # Save cluster centers
    centers_filename = "scikit_centers_{}.csv".format(dataset_name)
    centers_path = os.path.join(centers_dir, centers_filename)
    centers_df = pd.DataFrame(
        kmeans.cluster_centers_, 
        columns=['f{}'.format(i) for i in range(kmeans.cluster_centers_.shape[1])]
    )
    centers_df.to_csv(centers_path, index=False)
    print("Centers saved to: {}".format(centers_path))
    
    # Save labels
    labels_filename = "scikit_labels_{}.csv".format(dataset_name)
    labels_path = os.path.join(labels_dir, labels_filename)
    labels_df = pd.DataFrame(kmeans.labels_, columns=['label'])
    labels_df.to_csv(labels_path, index=False)
    print("Labels saved to: {}".format(labels_path))
    
    return kmeans, elapsed_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KMeans clustering")
    parser.add_argument("dataset", type=str, 
                        help="Path to dataset CSV")
    parser.add_argument("initialcluster", type=str, 
                        help="Path to initial centers CSV")
    parser.add_argument("clusters", type=int,
                        help="Number of clusters")
    parser.add_argument("iterations", type=int, default=3, nargs='?',
                        help="Maximum number of KMeans iterations")
    
    args = parser.parse_args()
    
    kmeans_model, runtime = run_kmeans(
        args.dataset, 
        args.initialcluster, 
        args.clusters,
        args.iterations
    )
    
    print("\nClustering completed successfully!")