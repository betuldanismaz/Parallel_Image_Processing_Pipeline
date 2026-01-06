# Image Processing Pipeline: Parallel Edge Detection

A high-performance image processing pipeline implementing 7×7 Sobel edge detection using three different parallelization strategies: serial baseline, OpenMP multi-threading, and MPI distributed processing. This project demonstrates performance optimization techniques for computationally intensive image processing tasks.

## Overview

This project implements a three-stage image processing pipeline:

1. **Preprocess Stage**: RGB to grayscale conversion
2. **Process Stage**: 7×7 Sobel edge detection via manual 2D convolution
3. **Postprocess Stage**: Binary thresholding for edge visualization

The pipeline is implemented in three parallel computing paradigms to analyze and compare performance characteristics across different parallelization approaches.

## Features

- **Serial Implementation**: Baseline single-threaded execution for performance comparison
- **OpenMP Implementation**: Shared-memory parallelization with dynamic scheduling and thread-safe energy accumulation
- **MPI Implementation**: Distributed-memory parallelization for multi-node execution
- **Automated Benchmarking**: CSV logging of execution metrics for performance analysis
- **7×7 Sobel Kernel**: Extended Sobel operators for enhanced edge detection accuracy
- **Energy Metrics**: Total gradient magnitude computation for image analysis


## Project Structure

```
pipeline_project/
│
├── CMakeLists.txt              # Build configuration
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore rules
├── input.jpg                   # Input image (user-provided)
│
├── serial/                     # Serial implementation
│   └── main.cpp
│
├── openmp/                     # OpenMP parallel implementation
│   └── main.cpp
│
├── mpi/                        # MPI distributed implementation
│   └── main.cpp
│
├── results/                    # Output directory
│   ├── output_serial_7x7.png
│   ├── output_openmp_7x7.png
│   ├── output_mpi_7x7.png
│   ├── openmp_benchmark_results.csv
│   ├── openmp_execution_time.png
│   ├── openmp_speedup.png
│   ├── mpi_benchmark_results.csv
│   ├── mpi_execution_time.png
│   └── mpi_speedup.png
│
├── analyze_results.py          # Benchmark analysis script (optional)
│
└── build/                      # CMake build artifacts (gitignored)
```


## Technical Specifications

### Sobel Edge Detection Algorithm

The implementation uses extended 7×7 Sobel kernels for horizontal (Gx) and vertical (Gy) gradient computation:

**Horizontal Gradient (Gx)**:

```
[-3 -2 -1  0  1  2  3]
[-4 -3 -2  0  2  3  4]
[-5 -4 -3  0  3  4  5]
[-6 -5 -4  0  4  5  6]
[-5 -4 -3  0  3  4  5]
[-4 -3 -2  0  2  3  4]
[-3 -2 -1  0  1  2  3]
```

**Vertical Gradient (Gy)**:

```
[-3 -4 -5 -6 -5 -4 -3]
[-2 -3 -4 -5 -4 -3 -2]
[-1 -2 -3 -4 -3 -2 -1]
[ 0  0  0  0  0  0  0]
[ 1  2  3  4  3  2  1]
[ 2  3  4  5  4  3  2]
[ 3  4  5  6  5  4  3]
```

**Gradient Magnitude**: |Gx| + |Gy|

### Parallelization Strategies

#### OpenMP Optimization

- **Directives**: `#pragma omp parallel for collapse(2) schedule(dynamic) reduction(+:totalEnergy)`
- **Loop Collapsing**: 2D loop fusion for improved thread utilization
- **Dynamic Scheduling**: Load balancing for irregular workloads
- **Reduction Clause**: Thread-safe accumulation of total energy

#### MPI Optimization

- Row-based domain decomposition for distributed processing
- Halo exchange for boundary region handling
- Collective operations for result gathering

## Requirements

### Dependencies

- **C++ Compiler**: GCC/Clang with C++11 support or MSVC
- **CMake**: Version 3.10 or higher
- **OpenCV**: Version 4.x (with core, imgproc, imgcodecs modules)
- **OpenMP**: Compiler support required
- **MPI**: MPICH, OpenMPI, or MS-MPI distribution

### Optional

- **Python 3.x**: For benchmark analysis (pandas, matplotlib)

## Build Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd pipeline_project
```

### 2. Create Build Directory

```bash
mkdir build
cd build
```

### 3. Configure with CMake

```bash
cmake ..
```

### 4. Compile

```bash
cmake --build .
```

Or using Make:

```bash
make
```

### Build Outputs

- `build/serial_app.exe` - Serial implementation
- `build/openmp_app.exe` - OpenMP parallel implementation
- `build/mpi_app.exe` - MPI distributed implementation

## Usage

### Prerequisites

Place an input image named `input.jpg` in the project root directory.

### Serial Execution

```bash
./build/serial_app.exe
```

**Output**:

- `results/output_serial_7x7.png` - Processed edge-detected image
- Console output with execution time

### OpenMP Execution

```bash
./build/openmp_app.exe
```

**Interactive Input**: The program will prompt for the number of threads:

```
Enter the number of threads to use: 8
```

**Outputs**:

- `results/output_openmp_7x7.png` - Processed image
- `results/openmp_benchmark_results.csv` - Benchmark log (timestamp, thread count, execution time, total energy)

### MPI Execution

```bash
mpiexec -n <num_processes> ./build/mpi_app.exe
```

**Example** (4 processes):

```bash
mpiexec -n 4 ./build/mpi_app.exe
```

**Example** (32 processes):

```bash
mpiexec -n 32 ./build/mpi_app.exe
```

**Outputs**:

- `results/output_mpi_7x7.png` - Processed image (from rank 0)
- `results/mpi_benchmark_results.csv` - Benchmark log (timestamp, process count, execution time)



## Benchmark Results

### CSV Format

**OpenMP Benchmarks** (`results/openmp_benchmark_results.csv`):

```csv
Timestamp,Thread_Count,Execution_Time_ms,Total_Energy
2026-01-07 14:23:45,4,1234.56,8765432
```

**MPI Benchmarks** (`results/mpi_benchmark_results.csv`):

```csv
Timestamp,Process_Count,Execution_Time_ms
2026-01-07 14:25:10,8,987.65
```

### Analyzing Results

If `analyze_results.py` is available:

```bash
python analyze_results.py
```

This script can generate performance comparison plots and speedup analysis.

## Performance Considerations

### Optimization Techniques

- **Boundary Handling**: 3-pixel border exclusion to prevent out-of-bounds access
- **Loop Unrolling**: Kernel convolution optimized for cache locality
- **Memory Layout**: Contiguous memory access patterns for better cache performance
- **Workload Distribution**: Dynamic scheduling in OpenMP, domain decomposition in MPI

### Expected Speedup

- **OpenMP**: Near-linear speedup up to physical core count (diminishing returns with hyperthreading)
- **MPI**: Scales with process count, limited by communication overhead and problem size

### Profiling Tips

- Use larger input images (e.g., 4K, 8K) for more accurate performance measurements
- Disable background processes during benchmarking
- Run multiple iterations and compute average execution time
- Monitor CPU utilization to verify parallel efficiency

## Algorithm Complexity

- **Time Complexity**: O(N × M × K²) where:

  - N = image height
  - M = image width
  - K = kernel size (7)

- **Space Complexity**: O(N × M) for image buffers


## Future Enhancements

- CUDA/OpenCL GPU implementations
- Dynamic kernel size selection
- Multi-image batch processing
- Real-time video stream processing
- Additional edge detection algorithms (Canny, Prewitt)
- Adaptive thresholding in postprocess stage

## License

This project is developed for educational and research purposes.
