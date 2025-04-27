#!/bin/bash

# Configuration
KMEANS_EXECUTABLE="./kmeans_serial"
DATASET_DIR="../kmeans_datasets_csv"
INIT_DIR="../kmeans_inits"
OUTPUT_FILE="kmeans_performance_summary.txt"
MAX_ITERATIONS=30

# Check if executable exists
if [ ! -f "$KMEANS_EXECUTABLE" ]; then
    echo "Error: $KMEANS_EXECUTABLE not found!"
    exit 1
fi

# Check if dataset directory exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory $DATASET_DIR not found!"
    exit 1
fi

# Check if init directory exists
if [ ! -d "$INIT_DIR" ]; then
    echo "Error: Init directory $INIT_DIR not found!"
    exit 1
fi

# List directories for debugging
echo "Dataset files:"
ls -la "$DATASET_DIR"
echo ""
echo "Init files:"
ls -la "$INIT_DIR"
echo ""

# Create/overwrite output file with header
echo "# KMeans Performance Summary" > "$OUTPUT_FILE"
echo "# Generated on $(date)" >> "$OUTPUT_FILE"
echo "# Format: Filename | Dimensions | Num Clusters | Num Points | Elapsed Time (s) | SSE" >> "$OUTPUT_FILE"
echo "----------------------------------------------------------------------------------" >> "$OUTPUT_FILE"

# Process each dataset file (use find to avoid glob problems)
for dataset_file in $(find "$DATASET_DIR" -name "*.csv"); do
    # Extract filename without extension
    filename=$(basename "$dataset_file" .csv)
    
    echo "Processing dataset: $filename"
    
    # Check if matching init file exists
    init_file="$INIT_DIR/${filename}_init_seed1234.csv"
    echo "Looking for init file: $init_file"
    
    if [ ! -f "$init_file" ]; then
        echo "Warning: Matching init file not found, skipping..."
        continue
    fi
    
    echo "Processing $filename..."
    
    # Extract metadata from filename (assuming format like "blobs_10000x1024d_10c_seed1234")
    if [[ $filename =~ ([0-9]+)x([0-9]+)d_([0-9]+)c ]]; then
        num_points="${BASH_REMATCH[1]}"
        dimensions="${BASH_REMATCH[2]}"
        num_clusters="${BASH_REMATCH[3]}"
        echo "Extracted metadata: $num_points points, $dimensions dimensions, $num_clusters clusters"
    else
        echo "Warning: Could not parse metadata from filename $filename"
        num_points="unknown"
        dimensions="unknown"
        num_clusters="unknown"
    fi
    
    # If we successfully extracted num_clusters, use it, otherwise use default value 10
    if [ "$num_clusters" == "unknown" ]; then
        clusters_arg=10
    else
        clusters_arg=$num_clusters
    fi
    
    # Run kmeans and capture output (print command for debugging)
    echo "Running: $KMEANS_EXECUTABLE $dataset_file $init_file $clusters_arg $MAX_ITERATIONS"
    output=$("$KMEANS_EXECUTABLE" "$dataset_file" "$init_file" "$clusters_arg" "$MAX_ITERATIONS" 2>&1)
    
    # Print full output for debugging
    echo "Command output:"
    echo "$output"
    echo ""
    
    # Extract elapsed time and SSE from output
    elapsed_time=$(echo "$output" | grep "Elapsed time:" | awk '{print $3}')
    sse=$(echo "$output" | grep "Sum of Squared Errors" | awk '{print $5}')
    
    # Write results to output file
    echo "$filename | $dimensions | $num_clusters | $num_points | $elapsed_time | $sse" >> "$OUTPUT_FILE"
done

echo "Done! Results saved to $OUTPUT_FILE"

# Print summary report
echo ""
echo "Summary Report:"
cat "$OUTPUT_FILE"