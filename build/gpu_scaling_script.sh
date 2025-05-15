#!/bin/bash

# Configuration
KMEANS_EXECUTABLE="./kmeans"
DATASET_DIR="../datasets_100_clusters"
INIT_DIR="../datasets_100_clusters_init"
OUTPUT_FILE="kmeans_performance_summary_ablation.txt"
MAX_ITERATIONS=20
USE_GPU=1          # Set to 1 to use GPU acceleration
USE_TRIANGLE=0     # Set to 1 to use Triangle Inequality optimization

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
echo "# Format: Filename | Dimensions | Num Clusters | Num Points | Algorithm Time (s) | Algorithm Time with GPU Warmup (s) | Iterations" >> "$OUTPUT_FILE"
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
    
    # Run kmeans with GPU flag and Triangle Inequality flag
    echo "Running: $KMEANS_EXECUTABLE $dataset_file $init_file $clusters_arg $MAX_ITERATIONS $USE_GPU $USE_TRIANGLE"
    output=$("$KMEANS_EXECUTABLE" "$dataset_file" "$init_file" "$clusters_arg" "$MAX_ITERATIONS" "$USE_GPU" "$USE_TRIANGLE" 2>&1)
    run_status=$?
    
    if [ $run_status -ne 0 ]; then
        echo "Error: Command failed with status $run_status"
        echo "Output:"
        echo "$output"
        continue
    fi
    
    # Extract algorithm time without GPU loading
    algo_time=$(echo "$output" | grep -E "^Algorithm time: " | awk '{print $3}')
    
    # Extract algorithm time including warmup
    algo_time_warmup=$(echo "$output" | grep -E "Algorithm time \(plus warmup\):" | awk '{print $5}')
    
    # For iterations, since we don't have it in the output, use max iterations
    iterations="$MAX_ITERATIONS"
    
    if [ -z "$algo_time" ]; then
        echo "Warning: Could not extract algorithm time from output"
        algo_time="N/A"
    fi
    
    if [ -z "$algo_time_warmup" ]; then
        echo "Warning: Could not extract algorithm time with GPU loading from output"
        algo_time_warmup="N/A"
    fi
    
    # Write results to output file
    echo "$filename | $dimensions | $num_clusters | $num_points | $algo_time | $algo_time_warmup | $iterations" >> "$OUTPUT_FILE"
    echo "Completed: Algorithm Time = $algo_time s, With GPU Loading = $algo_time_warmup s, Iterations = $iterations"
    echo "---------------------------------------------"
done

echo "Done! Results saved to $OUTPUT_FILE"
echo ""
echo "Summary Report:"
cat "$OUTPUT_FILE"