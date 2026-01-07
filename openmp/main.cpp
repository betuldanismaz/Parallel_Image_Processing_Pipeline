#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <omp.h>


// PREPROCESS STAGE: Convert RGB input to Grayscale (Serial)

void preprocess_stage(const cv::Mat& input, cv::Mat& gray) {
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
}


// PROCESS STAGE: Apply 7x7 Sobel Edge Detection with OpenMP Parallelization

void process_stage(const cv::Mat& gray, cv::Mat& output, long long& totalEnergy) {
    int rows = gray.rows;
    int cols = gray.cols;
    
    // Initialize output with zeros
    output = cv::Mat::zeros(gray.size(), CV_8UC1);
    
    // Initialize energy accumulator
    totalEnergy = 0;
    
    // Define 7x7 Extended Sobel Kernels (Horizontal and Vertical Gradients)
    const int kernel_size = 7;
    const int half_kernel = kernel_size / 2; // = 3
    
    // Gx (Horizontal Gradient) - 7x7 Sobel Kernel
    const int Gx[7][7] = {
        {-3, -2, -1,  0,  1,  2,  3},
        {-4, -3, -2,  0,  2,  3,  4},
        {-5, -4, -3,  0,  3,  4,  5},
        {-6, -5, -4,  0,  4,  5,  6},
        {-5, -4, -3,  0,  3,  4,  5},
        {-4, -3, -2,  0,  2,  3,  4},
        {-3, -2, -1,  0,  1,  2,  3}
    };
    
    // Gy (Vertical Gradient) - 7x7 Sobel Kernel
    const int Gy[7][7] = {
        {-3, -4, -5, -6, -5, -4, -3},
        {-2, -3, -4, -5, -4, -3, -2},
        {-1, -2, -3, -4, -3, -2, -1},
        { 0,  0,  0,  0,  0,  0,  0},
        { 1,  2,  3,  4,  3,  2,  1},
        { 2,  3,  4,  5,  4,  3,  2},
        { 3,  4,  5,  6,  5,  4,  3}
    };
    
    // OPENMP PARALLELIZATION: collapse(2), schedule(dynamic), and reduction
    // Generic 2D Convolution with OpenMP optimization
    #pragma omp parallel for collapse(2) schedule(dynamic) reduction(+:totalEnergy)
    for (int i = half_kernel; i < rows - half_kernel; ++i) {
        for (int j = half_kernel; j < cols - half_kernel; ++j) {
            
            int sumX = 0; // Accumulator for horizontal gradient
            int sumY = 0; // Accumulator for vertical gradient
            
            // Convolve 7x7 kernel with image neighborhood
            for (int ki = -half_kernel; ki <= half_kernel; ++ki) {
                for (int kj = -half_kernel; kj <= half_kernel; ++kj) {
                    
                    // Get pixel value from neighborhood
                    int pixel = gray.at<uchar>(i + ki, j + kj);
                    
                    // Map kernel indices: ki,kj ∈ [-3,3] → array indices [0,6]
                    int kernel_row = ki + half_kernel;
                    int kernel_col = kj + half_kernel;
                    
                    // Multiply pixel by kernel weights and accumulate
                    sumX += pixel * Gx[kernel_row][kernel_col];
                    sumY += pixel * Gy[kernel_row][kernel_col];
                }
            }
            
            // Compute gradient magnitude: |Gx| + |Gy|
            int magnitude = std::abs(sumX) + std::abs(sumY);
            
            // Accumulate total energy (reduction handles thread-safety)
            totalEnergy += magnitude;
            
            // Clamp to valid pixel range [0, 255]
            output.at<uchar>(i, j) = (magnitude > 255) ? 255 : static_cast<uchar>(magnitude);
        }
    }
}

// POSTPROCESS STAGE: Convert to Binary Image using Thresholding (Serial)
void postprocess_stage(const cv::Mat& result, cv::Mat& output) {
    // Apply binary threshold: pixels > 50 become white (255), others black (0)
    cv::threshold(result, output, 50, 255, cv::THRESH_BINARY);
}

// MAIN: OpenMP Parallel Pipeline with 7x7 Sobel Kernel
int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "OpenMP Image Processing Pipeline (7x7 Sobel)" << std::endl;
    std::cout << "==================================================" << std::endl;
    
    // GET THREAD COUNT FROM USER
    int threadCount;
    std::cout << "Enter the number of threads to use: ";
    std::cin >> threadCount;
    
    // Validate and set thread count
    if (threadCount < 1) {
        std::cerr << "ERROR: Thread count must be at least 1. Using default." << std::endl;
        threadCount = omp_get_max_threads();
    } else {
        omp_set_num_threads(threadCount);
    }
    
    std::cout << "Thread Count: " << threadCount << std::endl;
    
    // LOAD INPUT IMAGE
    std::string imagePath = "input.jpg";
    cv::Mat inputImage = cv::imread(imagePath, cv::IMREAD_COLOR);
    
    if (inputImage.empty()) {
        std::cerr << "ERROR: Could not open or find the image: " << imagePath << std::endl;
        return -1;
    }
    
    std::cout << "Image loaded: " << imagePath << " (" << inputImage.cols << "x" << inputImage.rows << " pixels)" << std::endl;
    
    // DECLARE PIPELINE VARIABLES
    cv::Mat grayImage;
    cv::Mat processedImage;
    cv::Mat outputImage;
    long long totalEnergy = 0;
    
    // START TIMING (Entire Pipeline)
    double startTime = static_cast<double>(cv::getTickCount());
    
    // ------------------------------------------------------------------------
    // STAGE 1: PREPROCESS (Serial)
    // ------------------------------------------------------------------------
    std::cout << "\n[STAGE 1] PREPROCESS: Converting to Grayscale..." << std::endl;
    preprocess_stage(inputImage, grayImage);
    std::cout << "[STAGE 1] PREPROCESS: Completed." << std::endl;
    
    // ------------------------------------------------------------------------
    // STAGE 2: PROCESS (Parallel with OpenMP)
    // ------------------------------------------------------------------------
    std::cout << "\n[STAGE 2] PROCESS: Applying 7x7 Sobel Edge Detection (OpenMP Parallel)..." << std::endl;
    process_stage(grayImage, processedImage, totalEnergy);
    std::cout << "[STAGE 2] PROCESS: Completed." << std::endl;
    std::cout << "[STAGE 2] Total Energy: " << totalEnergy << std::endl;
    
    // ------------------------------------------------------------------------
    // STAGE 3: POSTPROCESS (Serial)
    // ------------------------------------------------------------------------
    std::cout << "\n[STAGE 3] POSTPROCESS: Applying Binary Threshold..." << std::endl;
    postprocess_stage(processedImage, outputImage);
    std::cout << "[STAGE 3] POSTPROCESS: Completed." << std::endl;
    
    // STOP TIMING
    double endTime = static_cast<double>(cv::getTickCount());
    double executionTime = (endTime - startTime) * 1000.0 / cv::getTickFrequency();
    
    cv::imwrite("results/output_openmp_7x7.png", outputImage);
    
    // DISPLAY RESULTS
    std::cout << "\n==================================================" << std::endl;
    std::cout << "Total Pipeline Time: " << executionTime << " ms" << std::endl;
    std::cout << "Output saved as: results/output_openmp_7x7.png" << std::endl;
    std::cout << "==================================================" << std::endl;
    
    // LOG RESULTS TO CSV (Automated Benchmarking)
   
    std::string csvFilename = "results/openmp_benchmark_results.csv";
    bool fileExists = false;
    
    // Check if file exists by attempting to open in read mode
    std::ifstream checkFile(csvFilename);
    if (checkFile.good()) {
        fileExists = true;
    }
    checkFile.close();
    
    // Open CSV file in append mode
    std::ofstream csvFile(csvFilename, std::ios::app);
    
    if (!csvFile.is_open()) {
        std::cerr << "WARNING: Could not open benchmark_results.csv for writing." << std::endl;
    } else {
        // Write header if file is new
        if (!fileExists) {
            csvFile << "Timestamp,Thread_Count,Execution_Time_ms,Total_Energy" << std::endl;
        }
        
        // Get current timestamp
        auto now = std::chrono::system_clock::now();
        auto now_time_t = std::chrono::system_clock::to_time_t(now);
        std::tm now_tm;
        localtime_s(&now_tm, &now_time_t); // Windows-safe version
        
        std::ostringstream timestamp;
        timestamp << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S");
        
        // Write benchmark data
        csvFile << timestamp.str() << ","
                << threadCount << ","
                << std::fixed << std::setprecision(2) << executionTime << ","
                << totalEnergy << std::endl;
        
        csvFile.close();
        std::cout << "\nBenchmark results appended to: " << csvFilename << std::endl;
    }
    
    return 0;
}