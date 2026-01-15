# Parallel Image Processing Pipeline

## 7×7 Sobel Edge Detection with Serial, OpenMP, and MPI Implementations

[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![OpenMP](https://img.shields.io/badge/OpenMP-Enabled-orange.svg)](https://www.openmp.org/)
[![MPI](https://img.shields.io/badge/MPI-Distributed-red.svg)](https://www.mpi-forum.org/)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Technical Specifications](#technical-specifications)
5. [Requirements](#requirements)
6. [Installation & Build](#installation--build)
7. [Usage](#usage)
8. [Benchmark Results](#benchmark-results)
9. [Analysis Tools](#analysis-tools)
10. [Performance Metrics](#performance-metrics)
11. [Correctness Verification](#correctness-verification)
12. [Troubleshooting](#troubleshooting)
13. [License](#license)

---

## Introduction

This project implements a high-performance image processing pipeline for edge detection using **7×7 extended Sobel operators**. The pipeline is developed in three parallelization paradigms to demonstrate and compare performance characteristics:

| Implementation | Parallelization Model | Target Architecture |
| -------------- | --------------------- | ------------------- |
| **Serial**     | Single-threaded       | Baseline reference  |
| **OpenMP**     | Shared-memory         | Multi-core CPUs     |
| **MPI**        | Distributed-memory    | Multi-node clusters |

The project includes automated benchmarking, CSV logging, and Python analysis scripts for comprehensive performance evaluation.

---

## Features

### Core Functionality

- **Three-Stage Pipeline Architecture**
  - **Preprocess**: RGB to grayscale conversion using OpenCV
  - **Process**: Manual 7×7 Sobel convolution with gradient magnitude computation
  - **Postprocess**: Binary thresholding for edge visualization

### Parallelization Features

- **OpenMP Implementation**
  - Loop collapsing with `collapse(2)` for 2D parallelization
  - Dynamic scheduling for load balancing
  - Thread-safe energy accumulation via `reduction` clause
- **MPI Implementation**
  - Row-based domain decomposition
  - Ghost row exchange for boundary handling
  - `MPI_Scatterv` / `MPI_Gatherv` for efficient data distribution
  - `MPI_Reduce` for global energy computation

### Benchmarking & Analysis

- Automated CSV logging with timestamps
- Speedup and efficiency computation
- Comparative visualization (PNG/PDF plots)
- MSE-based correctness verification

---

## Project Structure

```
pipeline_project/
│
├── CMakeLists.txt                     # CMake build configuration
├── README.md                          # Project documentation (this file)
├── .gitignore                         # Git ignore rules
├── input.jpg                          # Input image (user-provided)
│
├── serial/                            # Serial Implementation
│   └── main.cpp                       # Single-threaded baseline pipeline
│
├── openmp/                            # OpenMP Implementation
│   └── main.cpp                       # Shared-memory parallel pipeline
│
├── mpi/                               # MPI Implementation
│   └── main.cpp                       # Distributed-memory parallel pipeline
│
├── scripts/                           # Python Analysis Scripts
│   ├── compare_openmp_mpi.py          # Generate comparison plots
│   ├── compute_efficiency.py          # Compute/update efficiency metrics
│   └── compute_mse.py                 # MSE computation and diff images
│
├── results/                           # Output Directory
│   ├── output_serial_7x7.png          # Serial output image
│   ├── output_openmp_7x7.png          # OpenMP output image
│   ├── output_mpi_7x7.png             # MPI output image
│   ├── openmp_benchmark_results.csv   # OpenMP benchmark logs
│   ├── mpi_benchmark_results.csv      # MPI benchmark logs
│   ├── compare_openmp_mpi.png         # Comparison plot (generated)
│   └── diff_serial_*.png              # Difference images (generated)
│
└── build/                             # Build artifacts (gitignored)
    ├── serial_app.exe
    ├── openmp_app.exe
    └── mpi_app.exe
```

### File Descriptions

| File/Directory                  | Description                                                                                             |
| ------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `serial/main.cpp`               | Baseline single-threaded implementation. Produces reference output and timing for speedup calculations. |
| `openmp/main.cpp`               | OpenMP parallelized version. Prompts for thread count at runtime and logs results to CSV.               |
| `mpi/main.cpp`                  | MPI distributed version. Uses rank 0 as master for I/O; all ranks participate in computation.           |
| `scripts/compare_openmp_mpi.py` | Reads benchmark CSVs and generates execution time and speedup comparison plots.                         |
| `scripts/compute_efficiency.py` | Computes `Efficiency (%) = (Speedup / P) × 100` and updates CSV files.                                  |
| `scripts/compute_mse.py`        | Computes Mean Squared Error between serial and parallel outputs for correctness verification.           |

---

## Technical Specifications

### Sobel Edge Detection Algorithm

The implementation uses extended **7×7 Sobel kernels** for improved edge detection accuracy compared to standard 3×3 kernels.

#### Horizontal Gradient Kernel (Gx)

```
┌────────────────────────────────────────┐
│  -3  -2  -1   0   1   2   3           │
│  -4  -3  -2   0   2   3   4           │
│  -5  -4  -3   0   3   4   5           │
│  -6  -5  -4   0   4   5   6           │
│  -5  -4  -3   0   3   4   5           │
│  -4  -3  -2   0   2   3   4           │
│  -3  -2  -1   0   1   2   3           │
└────────────────────────────────────────┘
```

#### Vertical Gradient Kernel (Gy)

```
┌────────────────────────────────────────┐
│  -3  -4  -5  -6  -5  -4  -3           │
│  -2  -3  -4  -5  -4  -3  -2           │
│  -1  -2  -3  -4  -3  -2  -1           │
│   0   0   0   0   0   0   0           │
│   1   2   3   4   3   2   1           │
│   2   3   4   5   4   3   2           │
│   3   4   5   6   5   4   3           │
└────────────────────────────────────────┘
```

#### Gradient Magnitude Computation

```
Magnitude(x,y) = |Gx(x,y)| + |Gy(x,y)|
```

### Algorithm Complexity

| Metric | Complexity                                               |
| ------ | -------------------------------------------------------- |
| Time   | O(N × M × K²) where N=height, M=width, K=kernel size (7) |
| Space  | O(N × M) for image buffers                               |

### Parallelization Strategies

#### OpenMP Strategy

```cpp
#pragma omp parallel for collapse(2) schedule(dynamic) reduction(+:totalEnergy)
for (int i = half_kernel; i < rows - half_kernel; ++i) {
    for (int j = half_kernel; j < cols - half_kernel; ++j) {
        // Convolution computation
    }
}
```

#### MPI Strategy

1. **Rank 0**: Load image, preprocess (grayscale conversion)
2. **Broadcast**: Image dimensions to all ranks
3. **Scatter**: Distribute image chunks with ghost rows (3 rows overlap)
4. **Compute**: Each rank processes its local chunk
5. **Reduce**: Aggregate total energy to rank 0
6. **Gather**: Collect processed chunks to rank 0
7. **Rank 0**: Postprocess (thresholding) and save output

---

## Requirements

### C++ Dependencies

| Dependency   | Version        | Purpose                                                        |
| ------------ | -------------- | -------------------------------------------------------------- |
| C++ Compiler | C++17 or later | GCC, Clang, or MSVC with OpenMP support                        |
| CMake        | 3.10+          | Build system                                                   |
| OpenCV       | 4.x            | Image I/O and preprocessing                                    |
| OpenMP       | 4.5+           | Shared-memory parallelization                                  |
| MPI          | 3.0+           | Distributed-memory parallelization (OpenMPI, MPICH, or MS-MPI) |

### Python Dependencies (Optional)

| Package    | Purpose                |
| ---------- | ---------------------- |
| pandas     | CSV data manipulation  |
| matplotlib | Plot generation        |
| numpy      | Numerical computations |

Install Python dependencies:

```bash
pip install pandas matplotlib numpy
```

---


### Build Outputs

| Executable   | Description                    |
| ------------ | ------------------------------ |
| `serial_app` | Serial baseline implementation |
| `openmp_app` | OpenMP parallel implementation |
| `mpi_app`    | MPI distributed implementation |

---

## Usage

### Prerequisites

Place an input image named `input.jpg` in the project root directory before running any executable.

### Serial Execution

```bash
./build/serial_app
```

**Output:**

- `results/output_serial_7x7.png` — Processed edge-detected image
- Console output with execution time

### OpenMP Execution

```bash
./build/openmp_app
```

**Interactive Prompt:**

```
Enter the number of threads to use: 8
```

**Output:**

- `results/output_openmp_7x7.png` — Processed image
- `results/openmp_benchmark_results.csv` — Benchmark log (appended)

### MPI Execution

```bash
mpiexec -n <num_processes> ./build/mpi_app
```

**Examples:**

```bash
# 4 MPI processes
mpiexec -n 4 ./build/mpi_app

# 16 MPI processes with process binding
mpiexec -n 16 --bind-to core ./build/mpi_app

# 32 MPI processes
mpiexec -n 32 ./build/mpi_app
```

**Output:**

- `results/output_mpi_7x7.png` — Processed image (from rank 0)
- `results/mpi_benchmark_results.csv` — Benchmark log (appended)

---

## Benchmark Results

### CSV Format

#### OpenMP Benchmarks (`results/openmp_benchmark_results.csv`)

| Column            | Description                                   |
| ----------------- | --------------------------------------------- |
| Timestamp         | Date and time of execution                    |
| Thread_Count      | Number of OpenMP threads used                 |
| Execution_Time_ms | Total pipeline execution time in milliseconds |
| Speedup           | T_serial / T_parallel                         |
| Efficiency        | (Speedup / Thread_Count) × 100%               |

#### MPI Benchmarks (`results/mpi_benchmark_results.csv`)

| Column            | Description                                   |
| ----------------- | --------------------------------------------- |
| Timestamp         | Date and time of execution                    |
| Rank_Count        | Number of MPI processes used                  |
| Execution_Time_ms | Total pipeline execution time in milliseconds |
| Speedup           | T_serial / T_parallel                         |
| Efficiency        | (Speedup / Rank_Count) × 100%                 |

### Sample Results

#### OpenMP Performance

| Threads | Execution Time (ms) | Speedup | Efficiency (%) |
| ------- | ------------------- | ------- | -------------- |
| 1       | 9677.71             | 0.94    | 93.96          |
| 2       | 5265.49             | 1.73    | 86.35          |
| 4       | 2894.76             | 3.14    | 78.53          |
| 8       | 1799.09             | 5.05    | 63.18          |
| 16      | 1296.61             | 7.01    | 43.83          |
| 32      | 1168.64             | 7.78    | 24.32          |

#### MPI Performance

| Ranks | Execution Time (ms) | Speedup | Efficiency (%) |
| ----- | ------------------- | ------- | -------------- |
| 1     | 6834.06             | 1.33    | 133.05\*       |
| 2     | 3716.01             | 2.45    | 122.35\*       |
| 4     | 2077.29             | 4.38    | 109.43\*       |
| 8     | 1236.55             | 7.35    | 91.92          |
| 16    | 819.13              | 11.10   | 69.38          |
| 32    | 789.65              | 11.52   | 35.99          |

> \*Note: Efficiency > 100% indicates super-linear speedup, typically caused by cache effects or baseline measurement inconsistencies.

---

## Analysis Tools

### Generate Comparison Plots

```bash
python scripts/compare_openmp_mpi.py
```

**Output:**

- `results/compare_openmp_mpi.png` — Side-by-side execution time and speedup plots
- `results/compare_openmp_mpi.pdf` — PDF version

### Compute/Update Efficiency

```bash
# Use existing Speedup values
python scripts/compute_efficiency.py

# Recompute Speedup from a specific serial baseline (in ms)
python scripts/compute_efficiency.py --baseline 9500
```

### Verify Correctness (MSE Computation)

```bash
python scripts/compute_mse.py
```

**Output:**

- Console: MSE values for each parallel implementation vs serial
- `results/diff_serial_openmp.png` — Visual difference heatmap
- `results/diff_serial_mpi.png` — Visual difference heatmap

---

## Performance Metrics

### Speedup

$$
S(P) = \frac{T_{serial}}{T_{parallel}(P)}
$$

Where:

- $T_{serial}$ = Execution time of serial implementation
- $T_{parallel}(P)$ = Execution time with P threads/processes

### Efficiency

$$
E(P) = \frac{S(P)}{P} \times 100\%
$$

Where:

- $S(P)$ = Speedup with P threads/processes
- $P$ = Number of threads/processes

### Ideal vs Observed Scaling

| Scaling Type     | Characteristic                                       |
| ---------------- | ---------------------------------------------------- |
| **Linear**       | Speedup = P, Efficiency = 100%                       |
| **Sub-linear**   | Speedup < P, Efficiency < 100%                       |
| **Super-linear** | Speedup > P, Efficiency > 100% (rare, cache effects) |

---

## Correctness Verification

### Bitwise Exactness Test

To verify that parallel implementations produce identical results to the serial baseline:

```bash
python scripts/compute_mse.py
```

**Expected Output for Correct Implementation:**

```
OpenMP vs Serial — MSE: 0.000000
MPI vs Serial — MSE: 0.000000
```

An MSE of 0.0 indicates bitwise-identical output images.

### Visual Inspection

Compare output images visually:

- `results/output_serial_7x7.png`
- `results/output_openmp_7x7.png`
- `results/output_mpi_7x7.png`

---

## Troubleshooting

### Common Issues

| Issue                     | Solution                                                           |
| ------------------------- | ------------------------------------------------------------------ |
| `input.jpg not found`     | Place an input image in the project root                           |
| OpenMP not detected       | Ensure compiler supports OpenMP (`-fopenmp` for GCC)               |
| MPI initialization failed | Check MPI installation and `mpiexec` path                          |
| Efficiency > 100%         | Review baseline measurement; use `--baseline` flag for consistency |
| Build errors on Windows   | Use MS-MPI and ensure environment variables are set                |

### Performance Optimization Tips

1. **Use large images** (4K/8K) to ensure computation dominates I/O overhead
2. **Pin processes to cores** using `--bind-to core` (MPI) or `OMP_PROC_BIND=true` (OpenMP)
3. **Run multiple iterations** and compute average to reduce noise
4. **Disable background processes** during benchmarking
5. **Check CPU frequency scaling** — use performance governor for consistent results

---

## License

This project is developed for educational and research purposes.

---

## Authors

Parallel Computing Course Project — 2026

---

## Acknowledgments

- OpenCV Library for image processing primitives
- OpenMP API for shared-memory parallelization
- MPI Forum for distributed computing standards
