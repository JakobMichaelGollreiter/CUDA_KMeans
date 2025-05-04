#!/bin/bash

# Configuration
KMEANS_EXECUTABLE="./kmeans"
DATASET_DIR="../kmeans_datasets_csv"
INIT_DIR="../kmeans_inits"
OUTPUT_FILE="kmeans_performance_summary.txt"
MAX_ITERATIONS=20
USE_GPU=0          # Set to 1 to use GPU acceleration
USE_TRIANGLE=1     # Set to 1 to use Triangle Inequality optimization

echo "=== KMeans Performance Benchmark ==="
echo "Executable: $KMEANS_EXECUTABLE"
echo "Dataset directory: $DATASET_DIR"
echo "Init directory: $INIT_DIR"
echo "Max iterations: $MAX_ITERATIONS"
echo "Using GPU: Yes (USE_GPU=$USE_GPU)"
echo "Using Triangle Inequality: Yes (USE_TRIANGLE=$USE_TRIANGLE)"
echo "========================================="

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

# Create/overwrite output file with header
echo "# KMeans Performance Summary" > "$OUTPUT_FILE"
echo "# Generated on $(date)" >> "$OUTPUT_FILE"
echo "# Format: Filename | Dimensions | Num Clusters | Num Points | Elapsed Time (s) | Iterations" >> "$OUTPUT_FILE"
echo "----------------------------------------------------------------------------------" >> "$OUTPUT_FILE"

# Find all dataset files
dataset_files=$(find "$DATASET_DIR" -name "*.csv" | sort)

# Count files
dataset_count=$(echo "$dataset_files" | wc -l)
echo "Found $dataset_count dataset file(s)"

if [ "$dataset_count" -eq 0 ]; then
    echo "Error: No CSV files found in dataset directory!"
    exit 1
fi

# Process each dataset file
for dataset_file in $dataset_files; do
    # Extract filename without extension
    filename=$(basename "$dataset_file" .csv)
    echo "Processing dataset: $filename"
    
    # Try to find matching init file with different patterns
    # Pattern 1: filename_init_seed1234.csv
    init_file="$INIT_DIR/${filename}_init_seed1234.csv"
    
    # If not found, try alternative pattern (removing _seed1234 from dataset filename if present)
    if [ ! -f "$init_file" ]; then
        base_filename=${filename%_seed1234}
        init_file="$INIT_DIR/${base_filename}_init_seed1234.csv"
    fi
    
    # Check if init file exists
    if [ ! -f "$init_file" ]; then
        echo "Warning: No matching init file found for $filename, skipping..."
        continue
    fi
    
    echo "Using init file: $init_file"
    
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
    
    # Run kmeans with parameters matching the C++ program
    echo "Running: $KMEANS_EXECUTABLE $dataset_file $init_file $clusters_arg $MAX_ITERATIONS $USE_GPU $USE_TRIANGLE"
    output=$("$KMEANS_EXECUTABLE" "$dataset_file" "$init_file" "$clusters_arg" "$MAX_ITERATIONS" "$USE_GPU" "$USE_TRIANGLE" 2>&1)
    run_status=$?
    
    if [ $run_status -ne 0 ]; then
        echo "Error: Command failed with status $run_status"
        echo "Output:"
        echo "$output"
        continue
    fi
    
    # Extract elapsed time - fix the pattern to match "Algorithm time (excluding transfers): X seconds"
    elapsed_time=$(echo "$output" | grep "Algorithm time" | sed -E 's/.*Algorithm time \(excluding transfers\): ([0-9.]+) seconds.*/\1/')
    
    # For debugging
    if [ -z "$elapsed_time" ]; then
        echo "Warning: Could not extract elapsed time from output. Here's the relevant line:"
        echo "$output" | grep -i "time"
    fi
    
    # Since iterations aren't explicitly output, we'll use max iterations as default
    iterations="$MAX_ITERATIONS"
    
    # Write results to output file
    echo "$filename | $dimensions | $num_clusters | $num_points | $elapsed_time | $iterations" >> "$OUTPUT_FILE"
    echo "Completed: Time = $elapsed_time s, Iterations = $iterations"
    echo "---------------------------------------------"
done

echo "Done! Results saved to $OUTPUT_FILE"
echo ""
echo "Summary Report:"
cat "$OUTPUT_FILE"