import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Function to parse data from text files
def parse_performance_data(file_path, processor_type):
    with open(file_path, 'r') as file:
        data_str = file.read()
    
    lines = data_str.strip().split('\n')
    # Skip header lines
    data_lines = [line for line in lines if line.startswith('blobs_')]
    
    # Create lists to store parsed data
    dimensions = []
    num_points = []
    elapsed_time = []
    
    for line in data_lines:
        parts = line.split('|')
        if len(parts) >= 5:  # Ensure we have enough parts
            dimensions.append(int(parts[1].strip()))
            num_points.append(int(parts[3].strip()))
            elapsed_time.append(float(parts[4].strip()))
    
    # Create a DataFrame
    df = pd.DataFrame({
        'Dimensions': dimensions,
        'NumPoints': num_points,
        'ElapsedTime': elapsed_time,
        'Processor': processor_type
    })
    
    return df

# Main function
def main():
    # Parse data from both files
    gpu_df = parse_performance_data('timing_gpu_kmeans.txt', 'GPU')
    cpu_df = parse_performance_data('timing_serial_cpu_perlmutter.txt', 'CPU')
    
    # Combine the DataFrames
    combined_df = pd.concat([gpu_df, cpu_df])
    
    # Set up the plotting style
    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 10))
    
    # Create plots
    # 1. Performance by Number of Points
    plt.subplot(2, 2, 1)
    sns.lineplot(
        data=combined_df, 
        x='NumPoints', 
        y='ElapsedTime', 
        hue='Processor',
        style='Dimensions',
        markers=True,
        dashes=False
    )
    plt.title('KMeans Performance: Time vs. Dataset Size')
    plt.xlabel('Number of Points')
    plt.ylabel('Elapsed Time (s)')
    plt.grid(True)
    
    # 2. Speedup factor (CPU/GPU) by number of points
    speedup_df = pd.merge(
        cpu_df, 
        gpu_df, 
        on=['NumPoints', 'Dimensions'], 
        suffixes=('_CPU', '_GPU')
    )
    speedup_df['Speedup'] = speedup_df['ElapsedTime_CPU'] / speedup_df['ElapsedTime_GPU']
    
    plt.subplot(2, 2, 2)
    sns.lineplot(
        data=speedup_df, 
        x='NumPoints', 
        y='Speedup', 
        style='Dimensions',
        markers=True,
        dashes=False
    )
    plt.title('GPU Speedup Factor (CPU Time / GPU Time)')
    plt.xlabel('Number of Points')
    plt.ylabel('Speedup Factor')
    plt.grid(True)
    
    # 3. Bar plot comparing CPU vs GPU for each dimension
    plt.subplot(2, 2, 3)
    pivot_df = combined_df.pivot_table(
        index=['NumPoints', 'Processor'], 
        columns='Dimensions', 
        values='ElapsedTime'
    ).reset_index()
    
    dim_512_df = combined_df[combined_df['Dimensions'] == 512].copy()
    sns.barplot(
        data=dim_512_df,
        x='NumPoints',
        y='ElapsedTime',
        hue='Processor',
        palette='viridis'
    )
    plt.title('Performance Comparison for 512 Dimensions')
    plt.xlabel('Number of Points')
    plt.ylabel('Elapsed Time (s)')
    plt.xticks(rotation=45)
    
    # 4. Bar plot for 1024 dimensions
    plt.subplot(2, 2, 4)
    dim_1024_df = combined_df[combined_df['Dimensions'] == 1024].copy()
    sns.barplot(
        data=dim_1024_df,
        x='NumPoints',
        y='ElapsedTime',
        hue='Processor',
        palette='viridis'
    )
    plt.title('Performance Comparison for 1024 Dimensions')
    plt.xlabel('Number of Points')
    plt.ylabel('Elapsed Time (s)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('kmeans_performance_comparison.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()