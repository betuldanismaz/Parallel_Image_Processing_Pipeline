
import argparse
from pathlib import Path
import pandas as pd
import sys


def update_file(path: Path, baseline: float | None):
    df = pd.read_csv(path)

    # detect threads/ranks column
    if 'Thread_Count' in df.columns:
        df['Threads'] = df['Thread_Count'].astype(float)
    elif 'Rank_Count' in df.columns:
        df['Threads'] = df['Rank_Count'].astype(float)
    elif 'Threads' in df.columns:
        df['Threads'] = df['Threads'].astype(float)
    else:
        raise ValueError(f'No Thread_Count/Rank_Count/Threads column found in {path}')

    # ensure execution time numeric
    if 'Execution_Time_ms' not in df.columns:
        raise ValueError(f'Execution_Time_ms column missing in {path}')
    df['Execution_Time_ms'] = pd.to_numeric(df['Execution_Time_ms'], errors='coerce')

    # Recompute Speedup if baseline provided
    if baseline is not None:
        df['Speedup'] = baseline / df['Execution_Time_ms']
    else:
        if 'Speedup' not in df.columns:
            raise ValueError(f'Speedup column missing in {path} and no baseline provided')
        df['Speedup'] = pd.to_numeric(df['Speedup'], errors='coerce')

    # Compute Efficiency as percentage
    df['Efficiency'] = (df['Speedup'] / df['Threads']) * 100.0

    # Round Efficiency to 4 decimal places
    df['Efficiency'] = df['Efficiency'].round(4)

    # Overwrite CSV (preserve column order: put Efficiency at end)
    # If Efficiency existed, it will be replaced
    df.to_csv(path, index=False)
    print(f'Updated {path} — Efficiency (%) written')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Compute Efficiency (%) for benchmark CSVs')
    parser.add_argument('--files', nargs='*', default=None,
                        help='CSV files to update (default: results/openmp_benchmark_results.csv and results/mpi_benchmark_results.csv)')
    parser.add_argument('--baseline', type=float, default=None,
                        help='Optional serial baseline time in ms to recompute Speedup as baseline / Execution_Time_ms')
    args = parser.parse_args(argv)

    default_files = [Path('results/openmp_benchmark_results.csv'), Path('results/mpi_benchmark_results.csv')]
    files = [Path(p) for p in args.files] if args.files else default_files

    for f in files:
        if not f.exists():
            print(f'Warning: {f} not found — skipping', file=sys.stderr)
            continue
        try:
            update_file(f, args.baseline)
        except Exception as e:
            print(f'Error processing {f}: {e}', file=sys.stderr)


if __name__ == '__main__':
    main()
