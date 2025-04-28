Changes made here: keep all the data on gpu, do not transfer to cpu between iterations, triangle inequality


mkdir build
cd build
cmake ..
make


./kmeans <data_file.csv> <centroids_file.csv> <num_clusters> [max_iterations] 1 1


Make sure to add the 1 at the end to enable gpu
