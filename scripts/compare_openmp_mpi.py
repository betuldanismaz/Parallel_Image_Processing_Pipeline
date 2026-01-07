import pandas as pd
import matplotlib.pyplot as plt
import os


def main():
    # Read CSVs
    op_csv = os.path.join('results', 'openmp_benchmark_results.csv')
    mpi_csv = os.path.join('results', 'mpi_benchmark_results.csv')

    op = pd.read_csv(op_csv)
    mpi = pd.read_csv(mpi_csv)

    # Normalize column names and types
    if 'Thread_Count' in op.columns:
        op['Threads'] = op['Thread_Count'].astype(int)
    else:
        op['Threads'] = range(1, len(op) + 1)

    if 'Rank_Count' in mpi.columns:
        mpi['Threads'] = mpi['Rank_Count'].astype(int)
    else:
        mpi['Threads'] = range(1, len(mpi) + 1)

    # Ensure numeric
    op['Execution_Time_ms'] = pd.to_numeric(op['Execution_Time_ms'], errors='coerce')
    op['Speedup'] = pd.to_numeric(op['Speedup'], errors='coerce')
    mpi['Execution_Time_ms'] = pd.to_numeric(mpi['Execution_Time_ms'], errors='coerce')
    mpi['Speedup'] = pd.to_numeric(mpi['Speedup'], errors='coerce')

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Execution time plot
    axes[0].plot(op['Threads'], op['Execution_Time_ms'], 'o-', label='OpenMP')
    axes[0].plot(mpi['Threads'], mpi['Execution_Time_ms'], 's-', label='MPI')
    axes[0].set_xlabel('Threads / Ranks')
    axes[0].set_ylabel('Execution Time (ms)')
    axes[0].set_xscale('log', base=2)
    axes[0].grid(True, which='both', ls='--', alpha=0.5)
    axes[0].legend()

    # Speedup plot
    axes[1].plot(op['Threads'], op['Speedup'], 'o-', label='OpenMP')
    axes[1].plot(mpi['Threads'], mpi['Speedup'], 's-', label='MPI')
    axes[1].set_xlabel('Threads / Ranks')
    axes[1].set_ylabel('Speedup')
    axes[1].set_xscale('log', base=2)
    axes[1].grid(True, which='both', ls='--', alpha=0.5)
    axes[1].legend()

    plt.tight_layout()

    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    out_png = os.path.join('results', 'compare_openmp_mpi.png')
    out_pdf = os.path.join('results', 'compare_openmp_mpi.pdf')

    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)

    print(f'Saved: {out_png} and {out_pdf}')


if __name__ == '__main__':
    main()
