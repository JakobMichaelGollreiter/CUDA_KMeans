Changes made here: change from array of structures to structure of arrays. 


mkdir build
cd build
cmake ..
make


./kmeans <data_file.csv> <centroids_file.csv> <num_clusters> [max_iterations] 1 1


Make sure to add the 1 at the end to enable gpu and the other 1 for triangle inequality. 
