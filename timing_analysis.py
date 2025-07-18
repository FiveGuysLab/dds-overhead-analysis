#!/usr/bin/env python3
"""
Timing Analysis Script
Parses executor timing results and creates box and whisker charts for analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import os

def parse_timing_data(filename):
    """
    Parse timing data from the text file.
    
    Args:
        filename (str): Path to the timing data file
        
    Returns:
        list: List of timing values in nanoseconds
    """
    timing_data = []
    
    try:
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        timing_data.append(int(line))
                    except ValueError:
                        print(f"Warning: Skipping invalid line: {line}")
        
        print(f"Successfully parsed {len(timing_data)} timing values")
        return timing_data
    
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

def analyze_timing_data(timing_data):
    """
    Perform statistical analysis on timing data.
    
    Args:
        timing_data (list): List of timing values
        
    Returns:
        dict: Dictionary containing statistical measures
    """
    if not timing_data:
        return {}
    
    # Convert to numpy array for calculations
    data = np.array(timing_data)
    
    # Basic statistics
    stats = {
        'count': len(data),
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'q1': np.percentile(data, 25),
        'q3': np.percentile(data, 75),
        'iqr': np.percentile(data, 75) - np.percentile(data, 25)
    }
    
    # Convert to microseconds for readability
    stats_microseconds = {k: v / 1000 for k, v in stats.items()}
    
    return stats, stats_microseconds

def ensure_results_dir():
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir

def create_box_whisker_plot(timing_data, output_file='timing_analysis.png'):
    """
    Create a comprehensive box and whisker plot with additional visualizations.
    
    Args:
        timing_data (list): List of timing values
        output_file (str): Output filename for the plot
    """
    if not timing_data:
        print("No data to plot")
        return
    # Convert to microseconds for better readability
    timing_microseconds = [x / 1000 for x in timing_data]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Executor Timing Analysis', fontsize=16, fontweight='bold')
    
    # 1. Box and Whisker Plot
    box_plot = ax1.boxplot(timing_microseconds, patch_artist=True, 
                          boxprops=dict(facecolor='lightblue', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2),
                          flierprops=dict(marker='o', markerfacecolor='red', markersize=4))
    ax1.set_title('Box and Whisker Plot (Microseconds)', fontweight='bold')
    ax1.set_ylabel('Time (μs)')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text
    stats, stats_micro = analyze_timing_data(timing_data)
    stats_text = f"""Statistics (μs):
Mean: {stats_micro['mean']:.2f}
Median: {stats_micro['median']:.2f}
Std Dev: {stats_micro['std']:.2f}
Q1: {stats_micro['q1']:.2f}
Q3: {stats_micro['q3']:.2f}
IQR: {stats_micro['iqr']:.2f}
Min: {stats_micro['min']:.2f}
Max: {stats_micro['max']:.2f}"""
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', fontsize=9, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. Histogram
    # Use more bins and log scale for better visualization of the spread
    ax2.hist(timing_microseconds, bins=100, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_title('Distribution Histogram (Microseconds)', fontweight='bold')
    ax2.set_xlabel('Time (μs)')
    ax2.set_ylabel('Frequency')
    ax2.set_yscale('log')  # Use log scale to better see the distribution
    ax2.grid(True, alpha=0.3)
    
    # Add mean and median lines
    ax2.axvline(stats_micro['mean'], color='red', linestyle='--', 
                label=f'Mean: {stats_micro["mean"]:.2f} μs')
    ax2.axvline(stats_micro['median'], color='green', linestyle='--', 
                label=f'Median: {stats_micro["median"]:.2f} μs')
    ax2.legend()
    
    # 3. Cumulative Distribution
    sorted_data = np.sort(timing_microseconds)
    cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax3.plot(sorted_data, cumulative, linewidth=2, color='purple')
    ax3.set_title('Cumulative Distribution Function', fontweight='bold')
    ax3.set_xlabel('Time (μs)')
    ax3.set_ylabel('Cumulative Probability')
    ax3.grid(True, alpha=0.3)
    
    # 4. Focused histogram on the main distribution (excluding extreme outliers)
    # Filter out extreme outliers for better visualization of the main distribution
    q99 = np.percentile(timing_microseconds, 99)
    filtered_data = [x for x in timing_microseconds if x <= q99]
    
    ax4.hist(filtered_data, bins=150, alpha=0.7, color='lightgreen', edgecolor='black')
    ax4.set_title(f'Main Distribution (≤ 99th percentile: {q99:.2f} μs)', fontweight='bold')
    ax4.set_xlabel('Time (μs)')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    
    # Add percentiles for reference
    p50 = np.percentile(timing_microseconds, 50)
    p90 = np.percentile(timing_microseconds, 90)
    p95 = np.percentile(timing_microseconds, 95)
    
    ax4.axvline(p50, color='red', linestyle='--', alpha=0.8, label=f'50th: {p50:.2f} μs')
    ax4.axvline(p90, color='orange', linestyle='--', alpha=0.8, label=f'90th: {p90:.2f} μs')
    ax4.axvline(p95, color='purple', linestyle='--', alpha=0.8, label=f'95th: {p95:.2f} μs')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_file}")
    plt.close()

def create_box_whisker_plot_wo_2000(timing_data, output_file='timing_analysis_wo_2000.png'):
    """
    Create a box and whisker plot and related charts, skipping the first 2000 runs.
    """
    if not timing_data or len(timing_data) <= 2000:
        print("Not enough data to plot without first 2000 runs.")
        return
    timing_data_wo_2000 = timing_data[2000:]
    timing_microseconds = [x / 1000 for x in timing_data_wo_2000]
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Executor Timing Analysis (Without First 2000 Runs)', fontsize=16, fontweight='bold')
    # 1. Box and Whisker Plot
    box_plot = ax1.boxplot(timing_microseconds, patch_artist=True, 
                          boxprops=dict(facecolor='lightblue', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2),
                          flierprops=dict(marker='o', markerfacecolor='red', markersize=4))
    ax1.set_title('Box and Whisker Plot (Microseconds)', fontweight='bold')
    ax1.set_ylabel('Time (μs)')
    ax1.grid(True, alpha=0.3)
    # Add statistics text
    stats, stats_micro = analyze_timing_data(timing_data_wo_2000)
    stats_text = f"""Statistics (μs):\nMean: {stats_micro['mean']:.2f}\nMedian: {stats_micro['median']:.2f}\nStd Dev: {stats_micro['std']:.2f}\nQ1: {stats_micro['q1']:.2f}\nQ3: {stats_micro['q3']:.2f}\nIQR: {stats_micro['iqr']:.2f}\nMin: {stats_micro['min']:.2f}\nMax: {stats_micro['max']:.2f}"""
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', fontsize=9, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    # 2. Histogram
    ax2.hist(timing_microseconds, bins=100, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_title('Distribution Histogram (Microseconds)', fontweight='bold')
    ax2.set_xlabel('Time (μs)')
    ax2.set_ylabel('Frequency')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(stats_micro['mean'], color='red', linestyle='--', 
                label=f'Mean: {stats_micro["mean"]:.2f} μs')
    ax2.axvline(stats_micro['median'], color='green', linestyle='--', 
                label=f'Median: {stats_micro["median"]:.2f} μs')
    ax2.legend()
    # 3. Cumulative Distribution
    sorted_data = np.sort(timing_microseconds)
    cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax3.plot(sorted_data, cumulative, linewidth=2, color='purple')
    ax3.set_title('Cumulative Distribution Function', fontweight='bold')
    ax3.set_xlabel('Time (μs)')
    ax3.set_ylabel('Cumulative Probability')
    ax3.grid(True, alpha=0.3)
    # 4. Focused histogram on the main distribution (excluding extreme outliers)
    q99 = np.percentile(timing_microseconds, 99)
    filtered_data = [x for x in timing_microseconds if x <= q99]
    ax4.hist(filtered_data, bins=150, alpha=0.7, color='lightgreen', edgecolor='black')
    ax4.set_title(f'Main Distribution (≤ 99th percentile: {q99:.2f} μs)', fontweight='bold')
    ax4.set_xlabel('Time (μs)')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    p50 = np.percentile(timing_microseconds, 50)
    p90 = np.percentile(timing_microseconds, 90)
    p95 = np.percentile(timing_microseconds, 95)
    ax4.axvline(p50, color='red', linestyle='--', alpha=0.8, label=f'50th: {p50:.2f} μs')
    ax4.axvline(p90, color='orange', linestyle='--', alpha=0.8, label=f'90th: {p90:.2f} μs')
    ax4.axvline(p95, color='purple', linestyle='--', alpha=0.8, label=f'95th: {p95:.2f} μs')
    ax4.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_file}")
    plt.close()

def create_detailed_analysis(timing_data, output_file='detailed_analysis.txt'):
    """
    Create a detailed text analysis of the timing data.
    
    Args:
        timing_data (list): List of timing values
        output_file (str): Output filename for the analysis
    """
    if not timing_data:
        print("No data to analyze")
        return
    
    timing_microseconds = [x / 1000 for x in timing_data]
    
    stats, stats_micro = analyze_timing_data(timing_data)
    
    with open(output_file, 'w') as f:
        f.write("EXECUTOR TIMING ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total measurements: {stats['count']:,}\n\n")
        
        f.write("TIMING STATISTICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Mean:           {stats_micro['mean']:.2f} μs ({stats['mean']:>10.0f} ns)\n")
        f.write(f"Median:         {stats_micro['median']:.2f} μs ({stats['median']:>10.0f} ns)\n")
        f.write(f"Standard Dev:   {stats_micro['std']:.2f} μs ({stats['std']:>10.0f} ns)\n")
        f.write(f"Minimum:        {stats_micro['min']:.2f} μs ({stats['min']:>10.0f} ns)\n")
        f.write(f"Maximum:        {stats_micro['max']:.2f} μs ({stats['max']:>10.0f} ns)\n")
        f.write(f"Q1 (25th %):    {stats_micro['q1']:.2f} μs ({stats['q1']:>10.0f} ns)\n")
        f.write(f"Q3 (75th %):    {stats_micro['q3']:.2f} μs ({stats['q3']:>10.0f} ns)\n")
        f.write(f"IQR:            {stats_micro['iqr']:.2f} μs ({stats['iqr']:>10.0f} ns)\n\n")
        
        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        f.write("PERCENTILES\n")
        f.write("-" * 20 + "\n")
        for p in percentiles:
            value = np.percentile(timing_data, p)
            f.write(f"{p:2d}th percentile: {value/1000:>10.2f} μs ({value:>10.0f} ns)\n")
        
        f.write("\n")
        
        # Outlier analysis
        q1, q3 = stats['q1'], stats['q3']
        iqr = stats['iqr']
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = [x for x in timing_data if x < lower_bound or x > upper_bound]
        
        f.write("OUTLIER ANALYSIS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Lower bound (Q1 - 1.5*IQR): {lower_bound/1000:.2f} μs ({lower_bound:.0f} ns)\n")
        f.write(f"Upper bound (Q3 + 1.5*IQR): {upper_bound/1000:.2f} μs ({upper_bound:.0f} ns)\n")
        f.write(f"Number of outliers: {len(outliers):,} ({len(outliers)/len(timing_data)*100:.2f}%)\n")
        
        if outliers:
            f.write(f"Outlier range: {min(outliers)/1000:.2f} - {max(outliers)/1000:.2f} μs\n")
        
        f.write("\n")
        
        # Performance categories
        f.write("PERFORMANCE CATEGORIES\n")
        f.write("-" * 25 + "\n")
        fast_threshold = stats_micro['q1']  # Below Q1
        slow_threshold = stats_micro['q3']  # Above Q3
        
        fast_count = sum(1 for x in timing_microseconds if x < fast_threshold)
        slow_count = sum(1 for x in timing_microseconds if x > slow_threshold)
        normal_count = len(timing_microseconds) - fast_count - slow_count
        
        f.write(f"Fast (< Q1):    {fast_count:>8,} measurements ({fast_count/len(timing_data)*100:>6.2f}%)\n")
        f.write(f"Normal (Q1-Q3):  {normal_count:>8,} measurements ({normal_count/len(timing_data)*100:>6.2f}%)\n")
        f.write(f"Slow (> Q3):     {slow_count:>8,} measurements ({slow_count/len(timing_data)*100:>6.2f}%)\n")
    
    print(f"Detailed analysis saved as {output_file}")

def create_detailed_histograms(timing_data, output_file='detailed_histograms.png'):
    """
    Create detailed histogram views for better analysis of data spread.
    
    Args:
        timing_data (list): List of timing values
        output_file (str): Output filename for the plots
    """
    if not timing_data:
        print("No data to plot")
        return
    timing_microseconds = [x / 1000 for x in timing_data]
    fig, ((ax1, ax2), (ax3, ax4), (ax5, _)) = plt.subplots(3, 2, figsize=(18, 16))
    fig.suptitle('Detailed Timing Distribution Analysis', fontsize=16, fontweight='bold')
    # 1. Full range histogram with log scale
    ax1.hist(timing_microseconds, bins=200, alpha=0.7, color='lightblue', edgecolor='black')
    ax1.set_title('Full Distribution (Log Scale)', fontweight='bold')
    ax1.set_xlabel('Time (μs)')
    ax1.set_ylabel('Frequency')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    # 2. Focus on main distribution (up to 95th percentile)
    p95 = np.percentile(timing_microseconds, 95)
    main_data = [x for x in timing_microseconds if x <= p95]
    ax2.hist(main_data, bins=200, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_title(f'Main Distribution (≤ 95th percentile: {p95:.2f} μs)', fontweight='bold')
    ax2.set_xlabel('Time (μs)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    # Add key percentiles
    p25 = np.percentile(timing_microseconds, 25)
    p50 = np.percentile(timing_microseconds, 50)
    p75 = np.percentile(timing_microseconds, 75)
    for ax in [ax1, ax2]:
        ax.axvline(p25, color='orange', linestyle='--', alpha=0.8, label=f'Q1: {p25:.2f} μs')
        ax.axvline(p50, color='red', linestyle='--', alpha=0.8, label=f'Median: {p50:.2f} μs')
        ax.axvline(p75, color='purple', linestyle='--', alpha=0.8, label=f'Q3: {p75:.2f} μs')
        ax.legend()
    # 3. Focus on the core distribution (up to 75th percentile)
    core_data = [x for x in timing_microseconds if x <= p75]
    ax3.hist(core_data, bins=150, alpha=0.7, color='lightcoral', edgecolor='black')
    ax3.set_title(f'Core Distribution (≤ 75th percentile: {p75:.2f} μs)', fontweight='bold')
    ax3.set_xlabel('Time (μs)')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    # Add more detailed percentiles
    p10 = np.percentile(timing_microseconds, 10)
    ax3.axvline(p10, color='blue', linestyle='--', alpha=0.8, label=f'10th: {p10:.2f} μs')
    ax3.axvline(p25, color='orange', linestyle='--', alpha=0.8, label=f'25th: {p25:.2f} μs')
    ax3.axvline(p50, color='red', linestyle='--', alpha=0.8, label=f'50th: {p50:.2f} μs')
    ax3.legend()
    # 4. Outlier analysis - focus on the tail (upper 90th+)
    p90 = np.percentile(timing_microseconds, 90)
    tail_data = [x for x in timing_microseconds if x > p90]
    ax4.hist(tail_data, bins=100, alpha=0.7, color='gold', edgecolor='black')
    ax4.set_title(f'Outlier Distribution (> 90th percentile: {p90:.2f} μs)', fontweight='bold')
    ax4.set_xlabel('Time (μs)')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    p95 = np.percentile(timing_microseconds, 95)
    p99 = np.percentile(timing_microseconds, 99)
    p999 = np.percentile(timing_microseconds, 99.9)
    ax4.axvline(p95, color='orange', linestyle='--', alpha=0.8, label=f'95th: {p95:.2f} μs')
    ax4.axvline(p99, color='red', linestyle='--', alpha=0.8, label=f'99th: {p99:.2f} μs')
    ax4.axvline(p999, color='purple', linestyle='--', alpha=0.8, label=f'99.9th: {p999:.2f} μs')
    ax4.legend()
    # 5. Outlier analysis - focus on the lower 0-10th percentile
    p0 = np.percentile(timing_microseconds, 0)
    p10 = np.percentile(timing_microseconds, 10)
    low_data = [x for x in timing_microseconds if x <= p10]
    ax5.hist(low_data, bins=50, alpha=0.7, color='deepskyblue', edgecolor='black')
    ax5.set_title(f'Lower Outlier Distribution (≤ 10th percentile: {p10:.2f} μs)', fontweight='bold')
    ax5.set_xlabel('Time (μs)')
    ax5.set_ylabel('Frequency')
    ax5.grid(True, alpha=0.3)
    ax5.axvline(p0, color='black', linestyle='--', alpha=0.8, label=f'Min: {p0:.2f} μs')
    ax5.axvline(p10, color='blue', linestyle='--', alpha=0.8, label=f'10th: {p10:.2f} μs')
    ax5.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Detailed histograms saved as {output_file}")
    plt.close()

def create_time_series_plot(timing_data, output_file='time_series_analysis.png'):
    """
    Create a time series plot showing timing values vs run number.
    
    Args:
        timing_data (list): List of timing values
        output_file (str): Output filename for the plot
    """
    if not timing_data:
        print("No data to plot")
        return
    timing_microseconds = [x / 1000 for x in timing_data]
    run_numbers = list(range(1, len(timing_microseconds) + 1))
    fig, ((ax1, ax2), (ax3, ax4), (ax5, _)) = plt.subplots(3, 2, figsize=(18, 16))
    fig.suptitle('Timing Performance Over Runs', fontsize=16, fontweight='bold')
    # 1. Full time series plot
    ax1.plot(run_numbers, timing_microseconds, alpha=0.6, linewidth=0.5, color='blue')
    ax1.set_title('Timing Values vs Run Number (Full Range)', fontweight='bold')
    ax1.set_xlabel('Run Number')
    ax1.set_ylabel('Time (μs)')
    ax1.grid(True, alpha=0.3)
    window_size = min(100, len(timing_microseconds) // 10)
    if window_size > 1:
        moving_avg = pd.Series(timing_microseconds).rolling(window=window_size).mean()
        ax1.plot(run_numbers, moving_avg, color='red', linewidth=2, 
                label=f'Moving Average (window={window_size})')
        ax1.legend()
    # 2. Focus on main distribution (≤ 99th percentile)
    p99 = np.percentile(timing_microseconds, 99)
    filtered_data = [(i+1, val) for i, val in enumerate(timing_microseconds) if val <= p99]
    run_nums_filtered, values_filtered = zip(*filtered_data) if filtered_data else ([], [])
    ax2.plot(run_nums_filtered, values_filtered, alpha=0.6, linewidth=0.5, color='green')
    ax2.set_title(f'Timing Values vs Run Number (≤ 99th percentile: {p99:.2f} μs)', fontweight='bold')
    ax2.set_xlabel('Run Number')
    ax2.set_ylabel('Time (μs)')
    ax2.grid(True, alpha=0.3)
    if len(values_filtered) > window_size:
        moving_avg_filtered = pd.Series(values_filtered).rolling(window=window_size).mean()
        ax2.plot(run_nums_filtered, moving_avg_filtered, color='red', linewidth=2,
                label=f'Moving Average (window={window_size})')
        ax2.legend()
    # 3. Focus on core distribution (≤ 75th percentile)
    p75 = np.percentile(timing_microseconds, 75)
    core_data = [(i+1, val) for i, val in enumerate(timing_microseconds) if val <= p75]
    run_nums_core, values_core = zip(*core_data) if core_data else ([], [])
    ax3.plot(run_nums_core, values_core, alpha=0.6, linewidth=0.5, color='orange')
    ax3.set_title(f'Core Performance (≤ 75th percentile: {p75:.2f} μs)', fontweight='bold')
    ax3.set_xlabel('Run Number')
    ax3.set_ylabel('Time (μs)')
    ax3.grid(True, alpha=0.3)
    if len(values_core) > window_size:
        moving_avg_core = pd.Series(values_core).rolling(window=window_size).mean()
        ax3.plot(run_nums_core, moving_avg_core, color='red', linewidth=2,
                label=f'Moving Average (window={window_size})')
        ax3.legend()
    # 4. Outlier analysis - show only the extreme values (upper 90th+)
    p90 = np.percentile(timing_microseconds, 90)
    outlier_data = [(i+1, val) for i, val in enumerate(timing_microseconds) if val > p90]
    run_nums_outlier, values_outlier = zip(*outlier_data) if outlier_data else ([], [])
    ax4.scatter(run_nums_outlier, values_outlier, alpha=0.7, color='red', s=20)
    ax4.set_title(f'Outlier Performance (> 90th percentile: {p90:.2f} μs)', fontweight='bold')
    ax4.set_xlabel('Run Number')
    ax4.set_ylabel('Time (μs)')
    ax4.grid(True, alpha=0.3)
    p95 = np.percentile(timing_microseconds, 95)
    p99 = np.percentile(timing_microseconds, 99)
    ax4.axhline(p95, color='orange', linestyle='--', alpha=0.8, label=f'95th: {p95:.2f} μs')
    ax4.axhline(p99, color='purple', linestyle='--', alpha=0.8, label=f'99th: {p99:.2f} μs')
    ax4.legend()
    # 5. Outlier analysis - show only the lowest values (≤ 10th percentile)
    p10 = np.percentile(timing_microseconds, 10)
    low_data = [(i+1, val) for i, val in enumerate(timing_microseconds) if val <= p10]
    run_nums_low, values_low = zip(*low_data) if low_data else ([], [])
    ax5.scatter(run_nums_low, values_low, alpha=0.7, color='deepskyblue', s=20)
    ax5.set_title(f'Lower Outlier Performance (≤ 10th percentile: {p10:.2f} μs)', fontweight='bold')
    ax5.set_xlabel('Run Number')
    ax5.set_ylabel('Time (μs)')
    ax5.grid(True, alpha=0.3)
    p0 = np.percentile(timing_microseconds, 0)
    ax5.axhline(p0, color='black', linestyle='--', alpha=0.8, label=f'Min: {p0:.2f} μs')
    ax5.axhline(p10, color='blue', linestyle='--', alpha=0.8, label=f'10th: {p10:.2f} μs')
    ax5.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Time series analysis saved as {output_file}")
    plt.close()

def create_combined_analysis(timing_data, output_file='combined_analysis.png'):
    """
    Create a comprehensive analysis combining time series and distribution views.
    
    Args:
        timing_data (list): List of timing values
        output_file (str): Output filename for the plot
    """
    if not timing_data:
        print("No data to plot")
        return
    timing_microseconds = [x / 1000 for x in timing_data]
    run_numbers = list(range(1, len(timing_microseconds) + 1))
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 15))
    fig.suptitle('Comprehensive Timing Analysis', fontsize=16, fontweight='bold')
    
    # 1. Time series plot (top)
    ax1.plot(run_numbers, timing_microseconds, alpha=0.6, linewidth=0.5, color='blue')
    ax1.set_title('Timing Performance Over All Runs', fontweight='bold')
    ax1.set_ylabel('Time (μs)')
    ax1.grid(True, alpha=0.3)
    
    # Add moving average
    window_size = min(100, len(timing_microseconds) // 10)
    if window_size > 1:
        moving_avg = pd.Series(timing_microseconds).rolling(window=window_size).mean()
        ax1.plot(run_numbers, moving_avg, color='red', linewidth=2, 
                label=f'Moving Average (window={window_size})')
        ax1.legend()
    
    # Add percentile reference lines
    p50 = np.percentile(timing_microseconds, 50)
    p90 = np.percentile(timing_microseconds, 90)
    p95 = np.percentile(timing_microseconds, 95)
    ax1.axhline(p50, color='green', linestyle='--', alpha=0.8, label=f'Median: {p50:.2f} μs')
    ax1.axhline(p90, color='orange', linestyle='--', alpha=0.8, label=f'90th: {p90:.2f} μs')
    ax1.axhline(p95, color='red', linestyle='--', alpha=0.8, label=f'95th: {p95:.2f} μs')
    ax1.legend()
    
    # 2. Distribution histogram (middle)
    ax2.hist(timing_microseconds, bins=200, alpha=0.7, color='lightblue', edgecolor='black')
    ax2.set_title('Distribution of Timing Values', fontweight='bold')
    ax2.set_xlabel('Time (μs)')
    ax2.set_ylabel('Frequency')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Add percentile lines
    ax2.axvline(p50, color='red', linestyle='--', alpha=0.8, label=f'Median: {p50:.2f} μs')
    ax2.axvline(p90, color='orange', linestyle='--', alpha=0.8, label=f'90th: {p90:.2f} μs')
    ax2.axvline(p95, color='purple', linestyle='--', alpha=0.8, label=f'95th: {p95:.2f} μs')
    ax2.legend()
    
    # 3. Box plot (bottom)
    box_plot = ax3.boxplot(timing_microseconds, patch_artist=True, 
                          boxprops=dict(facecolor='lightgreen', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2),
                          flierprops=dict(marker='o', markerfacecolor='red', markersize=4))
    ax3.set_title('Box and Whisker Plot', fontweight='bold')
    ax3.set_ylabel('Time (μs)')
    ax3.grid(True, alpha=0.3)
    
    # Add statistics text
    stats, stats_micro = analyze_timing_data(timing_data)
    stats_text = f"""Statistics:
Mean: {stats_micro['mean']:.2f} μs
Median: {stats_micro['median']:.2f} μs
Std Dev: {stats_micro['std']:.2f} μs
IQR: {stats_micro['iqr']:.2f} μs"""
    
    ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, 
             verticalalignment='top', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Combined analysis saved as {output_file}")
    plt.close()

def plot_lower_outlier_histogram(timing_data, output_file='lower_outlier_histogram.png'):
    """
    Plot a histogram for the 0-10th percentile (lower outliers).
    """
    if not timing_data:
        print("No data to plot")
        return
    timing_microseconds = [x / 1000 for x in timing_data]
    p10 = np.percentile(timing_microseconds, 10)
    low_data = [x for x in timing_microseconds if x <= p10]
    plt.figure(figsize=(10, 6))
    plt.hist(low_data, bins=50, alpha=0.7, color='deepskyblue', edgecolor='black')
    plt.title(f'Lower Outlier Distribution (≤ 10th percentile: {p10:.2f} μs)', fontweight='bold')
    plt.xlabel('Time (μs)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    p0 = np.percentile(timing_microseconds, 0)
    plt.axvline(p0, color='black', linestyle='--', alpha=0.8, label=f'Min: {p0:.2f} μs')
    plt.axvline(p10, color='blue', linestyle='--', alpha=0.8, label=f'10th: {p10:.2f} μs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Lower outlier histogram saved as {output_file}")
    plt.close()

def plot_lower_outlier_timeseries(timing_data, output_file='lower_outlier_timeseries.png'):
    """
    Plot a time series for the 0-10th percentile (lower outliers).
    """
    if not timing_data:
        print("No data to plot")
        return
    timing_microseconds = [x / 1000 for x in timing_data]
    run_numbers = list(range(1, len(timing_microseconds) + 1))
    p10 = np.percentile(timing_microseconds, 10)
    low_data = [(i+1, val) for i, val in enumerate(timing_microseconds) if val <= p10]
    run_nums_low, values_low = zip(*low_data) if low_data else ([], [])
    plt.figure(figsize=(12, 6))
    plt.scatter(run_nums_low, values_low, alpha=0.7, color='deepskyblue', s=20)
    plt.title(f'Lower Outlier Performance (≤ 10th percentile: {p10:.2f} μs)', fontweight='bold')
    plt.xlabel('Run Number')
    plt.ylabel('Time (μs)')
    plt.grid(True, alpha=0.3)
    p0 = np.percentile(timing_microseconds, 0)
    plt.axhline(p0, color='black', linestyle='--', alpha=0.8, label=f'Min: {p0:.2f} μs')
    plt.axhline(p10, color='blue', linestyle='--', alpha=0.8, label=f'10th: {p10:.2f} μs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Lower outlier time series saved as {output_file}")
    plt.close()

def process_timing_file(timing_file_path, test_node_name):
    """
    Process a single timing file and save results in results/{test_node_name}/{timing_filename}/
    """
    timing_filename = os.path.splitext(os.path.basename(timing_file_path))[0]
    results_dir = os.path.join('results', test_node_name, timing_filename)
    os.makedirs(results_dir, exist_ok=True)

    # Parse timing data
    print(f"\nProcessing {timing_file_path} ...")
    timing_data = parse_timing_data(timing_file_path)
    if not timing_data:
        print(f"No valid timing data found in {timing_file_path}")
        return

    # Perform analysis
    stats, stats_micro = analyze_timing_data(timing_data)
    if not stats:
        print(f"Failed to analyze data in {timing_file_path}")
        return

    # Print summary
    print("="*50)
    print(f"TIMING ANALYSIS SUMMARY for {timing_filename}")
    print("="*50)
    print(f"Total measurements: {stats['count']:,}")
    print(f"Mean time: {stats_micro['mean']:.2f} μs ({stats['mean']:,.0f} ns)")
    print(f"Median time: {stats_micro['median']:.2f} μs ({stats['median']:,.0f} ns)")
    print(f"Standard deviation: {stats_micro['std']:.2f} μs ({stats['std']:,.0f} ns)")
    print(f"Range: {stats_micro['min']:.2f} - {stats_micro['max']:.2f} μs")
    print(f"IQR: {stats_micro['iqr']:.2f} μs")

    # Create visualizations and analysis in this results_dir
    create_box_whisker_plot(timing_data, output_file=os.path.join(results_dir, 'timing_analysis.png'))
    create_box_whisker_plot_wo_2000(timing_data, output_file=os.path.join(results_dir, 'timing_analysis_wo_2000.png'))
    create_detailed_analysis(timing_data, output_file=os.path.join(results_dir, 'detailed_analysis.txt'))
    create_detailed_histograms(timing_data, output_file=os.path.join(results_dir, 'detailed_histograms.png'))
    create_time_series_plot(timing_data, output_file=os.path.join(results_dir, 'time_series_analysis.png'))
    create_combined_analysis(timing_data, output_file=os.path.join(results_dir, 'combined_analysis.png'))
    # New: lower outlier plots
    plot_lower_outlier_histogram(timing_data, output_file=os.path.join(results_dir, 'lower_outlier_histogram.png'))
    plot_lower_outlier_timeseries(timing_data, output_file=os.path.join(results_dir, 'lower_outlier_timeseries.png'))
    print(f"Analysis complete for {timing_filename}! Results in {results_dir}")

def main():
    """Main function to run the timing analysis for all subdirectories in 'execution-logs' directory."""
    timing_dir = 'execution-logs'
    results_dir = 'results'
    
    if not os.path.exists(timing_dir):
        print(f"Timing directory '{timing_dir}' not found.")
        return
    
    # Find all subdirectories in execution-logs directory (excluding .git and other hidden dirs)
    execution_subdirs = [d for d in os.listdir(timing_dir) 
                        if os.path.isdir(os.path.join(timing_dir, d)) and not d.startswith('.')]
    
    if not execution_subdirs:
        print(f"No subdirectories found in '{timing_dir}'.")
        return
    
    # Check which test nodes already have results
    existing_results = set()
    if os.path.exists(results_dir):
        existing_results = set(d for d in os.listdir(results_dir) 
                              if os.path.isdir(os.path.join(results_dir, d)) and not d.startswith('.'))
        print(f"Found existing results for: {sorted(existing_results)}")
    
    # Only process test nodes that don't have results yet
    subdirs_to_process = [d for d in execution_subdirs if d not in existing_results]
    
    if not subdirs_to_process:
        print("All test nodes already have results. No new processing needed.")
        return
    
    print(f"Processing test nodes: {sorted(subdirs_to_process)}")
    
    for subdir in subdirs_to_process:
        subdir_path = os.path.join(timing_dir, subdir)
        print(f"\nProcessing test node: {subdir}")
        
        # Find all .txt files in subdirectory
        txt_files = [f for f in os.listdir(subdir_path) 
                     if f.endswith('.txt')]
        
        if not txt_files:
            print(f"No .txt files found in '{subdir_path}'.")
            continue
        
        # Process each .txt file
        for txt_file in txt_files:
            timing_file_path = os.path.join(subdir_path, txt_file)
            print(f"  Processing file: {txt_file}")
            process_timing_file(timing_file_path, subdir)

if __name__ == "__main__":
    main() 