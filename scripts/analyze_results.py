import pandas as pd
import matplotlib.pyplot as plt
import os


SERIAL_BASELINE_MS = 9093.02  # Fixed serial baseline execution time


OPENMP_FILE = "results/openmp_benchmark_results.csv"
MPI_FILE = "results/mpi_benchmark_results.csv"
RESULTS_DIR = "results"


def calculate_speedup(df, baseline_time):
 
    df['Speedup'] = baseline_time / df['Execution_Time_ms']
    return df

def process_csv_file(filepath, label):

    if not os.path.exists(filepath):
        print(f"⚠ {label} file not found: {filepath}")
        return None
    
    try:
        print(f"\n[{label}] Processing: {filepath}")
        
        # Load CSV
        df = pd.read_csv(filepath)
        
        # Validate required column
        if 'Execution_Time_ms' not in df.columns:
            print(f"  ❌ ERROR: Missing 'Execution_Time_ms' column!")
            return None
        
        print(f"  Records: {len(df)}")
        
        # Calculate speedup (no rounding)
        df = calculate_speedup(df, SERIAL_BASELINE_MS)
        
        # Save back to original file
        df.to_csv(filepath, index=False)
        print(f"  ✓ Speedup column added/updated")
        
        return df
        
    except Exception as e:
        print(f"  ❌ ERROR: {str(e)}")
        return None


def ensure_results_directory():
    """Create results directory if it doesn't exist."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"\n✓ Created directory: {RESULTS_DIR}")

def plot_openmp_charts(df):

    if df is None or df.empty:
        print("\n⚠ No OpenMP data available for plotting")
        return
    
    print("\n[OpenMP Visualization]")
    
    # Sort by Thread_Count for clean line plots
    df = df.sort_values('Thread_Count')
    
    # Chart 1: Thread Count vs Execution Time
    plt.figure(figsize=(10, 6))
    plt.plot(df['Thread_Count'], df['Execution_Time_ms'], 'o-', linewidth=2, markersize=8)
    
    # Set precise tick labels on both axes
    plt.xticks(df['Thread_Count'])
    plt.yticks(df['Execution_Time_ms'])
    
    plt.xlabel('Thread Count', fontsize=12)
    plt.ylabel('Execution Time (ms)', fontsize=12)
    plt.title('OpenMP: Thread Count vs Execution Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(RESULTS_DIR, 'openmp_execution_time.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  ✓ Saved: {output_path}")
    
    # Chart 2: Thread Count vs Speedup
    plt.figure(figsize=(10, 6))
    plt.plot(df['Thread_Count'], df['Speedup'], 'o-', linewidth=2, markersize=8, color='green')
    
    # Set precise tick labels on both axes
    plt.xticks(df['Thread_Count'])
    plt.yticks(df['Speedup'])
    
    plt.xlabel('Thread Count', fontsize=12)
    plt.ylabel('Speedup', fontsize=12)
    plt.title('OpenMP: Speedup Curve (Baseline: 9093.02ms)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(RESULTS_DIR, 'openmp_speedup.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  ✓ Saved: {output_path}")

def plot_mpi_charts(df):

    if df is None or df.empty:
        print("\n⚠ No MPI data available for plotting")
        return
    
    print("\n[MPI Visualization]")
    
    # Sort by Rank_Count for clean line plots
    df = df.sort_values('Rank_Count')
    
    # Chart 3: Rank Count vs Execution Time
    plt.figure(figsize=(10, 6))
    plt.plot(df['Rank_Count'], df['Execution_Time_ms'], 'o-', linewidth=2, markersize=8, color='red')
    
    # Set precise tick labels on both axes
    plt.xticks(df['Rank_Count'])
    plt.yticks(df['Execution_Time_ms'])
    # Rotate Y-axis labels for better readability
    plt.gca().tick_params(axis='y', labelsize=9)
    for label in plt.gca().get_yticklabels():
        label.set_rotation(0)
        label.set_horizontalalignment('right')
    
    plt.xlabel('Rank Count', fontsize=12)
    plt.ylabel('Execution Time (ms)', fontsize=12)
    plt.title('MPI: Rank Count vs Execution Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(RESULTS_DIR, 'mpi_execution_time.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  ✓ Saved: {output_path}")
    
    # Chart 4: Rank Count vs Speedup
    plt.figure(figsize=(10, 6))
    plt.plot(df['Rank_Count'], df['Speedup'], 'o-', linewidth=2, markersize=8, color='purple')
    
    # Set precise tick labels on both axes
    plt.xticks(df['Rank_Count'])
    plt.yticks(df['Speedup'])
    # Rotate Y-axis labels for better readability
    plt.gca().tick_params(axis='y', labelsize=9)
    for label in plt.gca().get_yticklabels():
        label.set_rotation(0)
        label.set_horizontalalignment('right')
    
    plt.xlabel('Rank Count', fontsize=12)
    plt.ylabel('Speedup', fontsize=12)
    plt.title('MPI: Speedup Curve (Baseline: 9093.02ms)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(RESULTS_DIR, 'mpi_speedup.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  ✓ Saved: {output_path}")

def print_summary(openmp_df, mpi_df):

    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    print(f"Serial Baseline: {SERIAL_BASELINE_MS} ms\n")
    
    if openmp_df is not None and not openmp_df.empty:
        max_speedup = openmp_df['Speedup'].max()
        best_row = openmp_df.loc[openmp_df['Speedup'].idxmax()]
        print(f"OpenMP:")
        print(f"  Max Speedup: {max_speedup:.4f}x")
        print(f"  Best Config: {int(best_row['Thread_Count'])} threads")
        print(f"  Time: {best_row['Execution_Time_ms']:.2f} ms\n")
    
    if mpi_df is not None and not mpi_df.empty:
        max_speedup = mpi_df['Speedup'].max()
        best_row = mpi_df.loc[mpi_df['Speedup'].idxmax()]
        print(f"MPI:")
        print(f"  Max Speedup: {max_speedup:.4f}x")
        print(f"  Best Config: {int(best_row['Rank_Count'])} ranks")
        print(f"  Time: {best_row['Execution_Time_ms']:.2f} ms\n")
    
    print("="*70)


def main():
    """Main execution function."""
    print("="*70)
    print("HPC BENCHMARK ANALYSIS & VISUALIZATION TOOL")
    print("="*70)
    print(f"Serial Baseline: {SERIAL_BASELINE_MS} ms")
    
    # Ensure results directory exists
    ensure_results_directory()
    
    # TASK 1: Process CSV files (Data Enrichment)
    print("\n" + "-"*70)
    print("TASK 1: DATA ENRICHMENT (Calculating Speedup)")
    print("-"*70)
    
    openmp_df = process_csv_file(OPENMP_FILE, "OpenMP")
    mpi_df = process_csv_file(MPI_FILE, "MPI")
    
    # TASK 2: Generate visualizations
    print("\n" + "-"*70)
    print("TASK 2: VISUALIZATION (Generating Charts)")
    print("-"*70)
    
    plot_openmp_charts(openmp_df)
    plot_mpi_charts(mpi_df)
    
    # Print summary
    if openmp_df is not None or mpi_df is not None:
        print_summary(openmp_df, mpi_df)
        print("\n✓ Analysis complete!")
        print(f"  - CSV files updated with Speedup column")
        print(f"  - Charts saved to '{RESULTS_DIR}/' directory\n")
    else:
        print("\n⚠ No benchmark files were processed.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Analysis interrupted by user.")
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
