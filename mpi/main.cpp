#include <opencv2/opencv.hpp>
#include <mpi.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <filesystem>


// PREPROCESS STAGE: Convert RGB input to Grayscale (Master only)

void preprocess_stage(const cv::Mat& input, cv::Mat& gray) {
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
}


// PROCESS STAGE: Apply 7x7 Sobel Edge Detection (MPI Distributed)

void process_stage(cv::Mat& grayImage, cv::Mat& outputImage, int rank, int size, 
                   long long& totalEnergy) {
    
    const int half_kernel = 3;
    int rows = 0, cols = 0;
    
    // Get dimensions
    if (rank == 0) {
        rows = grayImage.rows;
        cols = grayImage.cols;
    }
    
    // MPI_BCAST: Broadcast dimensions to all ranks
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Calculate chunk distribution with ghost rows
    int baseChunkSize = (rows - 2 * half_kernel) / size;
    int remainder = (rows - 2 * half_kernel) % size;
    
    std::vector<int> sendCounts(size);
    std::vector<int> displs(size);
    std::vector<int> recvCounts(size);
    std::vector<int> recvDispls(size);
    
    int currentRow = half_kernel;
    for (int i = 0; i < size; ++i) {
        int chunkSize = baseChunkSize + (i < remainder ? 1 : 0);
        
        // Add ghost rows (3 above, 3 below)
        int ghostTop = half_kernel;
        int ghostBottom = half_kernel;
        int totalChunkRows = chunkSize + ghostTop + ghostBottom;
        
        sendCounts[i] = totalChunkRows * cols;
        displs[i] = (currentRow - ghostTop) * cols;
        
        recvCounts[i] = chunkSize * cols;
        recvDispls[i] = (currentRow - half_kernel) * cols;
        
        currentRow += chunkSize;
    }
    
    // MPI_SCATTERV: Distribute image chunks with ghost rows
    std::vector<uchar> localGrayChunk(sendCounts[rank]);
    
    if (rank == 0) {
        MPI_Scatterv(grayImage.data, sendCounts.data(), displs.data(), MPI_UNSIGNED_CHAR,
                     localGrayChunk.data(), sendCounts[rank], MPI_UNSIGNED_CHAR,
                     0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_UNSIGNED_CHAR,
                     localGrayChunk.data(), sendCounts[rank], MPI_UNSIGNED_CHAR,
                     0, MPI_COMM_WORLD);
    }
    
    // Process local chunk with 7x7 Sobel
    int localChunkRows = sendCounts[rank] / cols;
    std::vector<uchar> localOutputChunk(localChunkRows * cols, 0);
    long long localEnergy = 0;
    
    // Define 7x7 Extended Sobel Kernels
    const int Gx[7][7] = {
        {-3, -2, -1,  0,  1,  2,  3},
        {-4, -3, -2,  0,  2,  3,  4},
        {-5, -4, -3,  0,  3,  4,  5},
        {-6, -5, -4,  0,  4,  5,  6},
        {-5, -4, -3,  0,  3,  4,  5},
        {-4, -3, -2,  0,  2,  3,  4},
        {-3, -2, -1,  0,  1,  2,  3}
    };
    
    const int Gy[7][7] = {
        {-3, -4, -5, -6, -5, -4, -3},
        {-2, -3, -4, -5, -4, -3, -2},
        {-1, -2, -3, -4, -3, -2, -1},
        { 0,  0,  0,  0,  0,  0,  0},
        { 1,  2,  3,  4,  3,  2,  1},
        { 2,  3,  4,  5,  4,  3,  2},
        { 3,  4,  5,  6,  5,  4,  3}
    };
    
    // Process the chunk (with ghost rows available)
    for (int i = half_kernel; i < localChunkRows - half_kernel; ++i) {
        for (int j = half_kernel; j < cols - half_kernel; ++j) {
            
            int sumX = 0;
            int sumY = 0;
            
            // Convolve 7x7 kernel
            for (int ki = -half_kernel; ki <= half_kernel; ++ki) {
                for (int kj = -half_kernel; kj <= half_kernel; ++kj) {
                    int pixel = localGrayChunk[(i + ki) * cols + (j + kj)];
                    int kernel_row = ki + half_kernel;
                    int kernel_col = kj + half_kernel;
                    
                    sumX += pixel * Gx[kernel_row][kernel_col];
                    sumY += pixel * Gy[kernel_row][kernel_col];
                }
            }
            
            int magnitude = std::abs(sumX) + std::abs(sumY);
            localEnergy += magnitude;
            localOutputChunk[i * cols + j] = (magnitude > 255) ? 255 : static_cast<uchar>(magnitude);
        }
    }
    
    // MPI_REDUCE: Sum total energy from all ranks
    MPI_Reduce(&localEnergy, &totalEnergy, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // MPI_GATHERV: Collect processed chunks back to master
    if (rank == 0) {
        outputImage = cv::Mat::zeros(rows, cols, CV_8UC1);
        MPI_Gatherv(localOutputChunk.data(), recvCounts[rank], MPI_UNSIGNED_CHAR,
                    outputImage.data, recvCounts.data(), recvDispls.data(), MPI_UNSIGNED_CHAR,
                    0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(localOutputChunk.data(), recvCounts[rank], MPI_UNSIGNED_CHAR,
                    nullptr, nullptr, nullptr, MPI_UNSIGNED_CHAR,
                    0, MPI_COMM_WORLD);
    }
}


// POSTPROCESS STAGE: Convert to Binary Image using Thresholding (Master only)

void postprocess_stage(const cv::Mat& result, cv::Mat& output) {
    cv::threshold(result, output, 50, 255, cv::THRESH_BINARY);
}


// MAIN: MPI Distributed Pipeline

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    cv::Mat inputImage, grayImage, processedImage, outputImage;
    long long totalEnergy = 0;
    double startTime, endTime;
    
   
    // RANK 0: Load Input Image
    
    if (rank == 0) {
        std::cout << "==================================================" << std::endl;
        std::cout << "MPI Image Processing Pipeline (7x7 Sobel)" << std::endl;
        std::cout << "==================================================" << std::endl;
        std::cout << "MPI Ranks: " << size << std::endl;
        
        std::string imagePath = "input.jpg";
        inputImage = cv::imread(imagePath, cv::IMREAD_COLOR);
        
        if (inputImage.empty()) {
            std::cerr << "ERROR: Could not open or find the image: " << imagePath << std::endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        
        std::cout << "Image loaded: " << imagePath << " (" 
                  << inputImage.cols << "x" << inputImage.rows << " pixels)" << std::endl;
    }
    
    startTime = MPI_Wtime();
    

    // STAGE 1: PREPROCESS (Master only)

    if (rank == 0) {
        std::cout << "\n[STAGE 1] PREPROCESS: Converting to Grayscale..." << std::endl;
        preprocess_stage(inputImage, grayImage);
        std::cout << "[STAGE 1] PREPROCESS: Completed." << std::endl;
    }
    

    // STAGE 2: PROCESS (MPI Distributed)

    if (rank == 0) {
        std::cout << "\n[STAGE 2] PROCESS: Applying 7x7 Sobel Edge Detection (MPI Distributed)..." << std::endl;
    }
    
    process_stage(grayImage, processedImage, rank, size, totalEnergy);
    
    if (rank == 0) {
        std::cout << "[STAGE 2] PROCESS: Completed." << std::endl;
        std::cout << "[STAGE 2] Total Energy: " << totalEnergy << std::endl;
    }
    

    // STAGE 3: POSTPROCESS (Master only)

    if (rank == 0) {
        std::cout << "\n[STAGE 3] POSTPROCESS: Applying Binary Threshold..." << std::endl;
        postprocess_stage(processedImage, outputImage);
        std::cout << "[STAGE 3] POSTPROCESS: Completed." << std::endl;
    }
    
    endTime = MPI_Wtime();
    double executionTime = (endTime - startTime) * 1000.0; // Convert to ms
    

    // RANK 0: Save Results and Log to CSV

    if (rank == 0) {
        cv::imwrite("output_mpi_7x7.png", outputImage);
        
        std::cout << "\n==================================================" << std::endl;
        std::cout << "Total Pipeline Time: " << executionTime << " ms" << std::endl;
        std::cout << "Output saved as: output_mpi_7x7.png" << std::endl;
        std::cout << "==================================================" << std::endl;
        

        // BENCHMARK LOGGING TO CSV

        std::string resultsDir = "results";
        
        // Create results directory if it doesn't exist
        if (!std::filesystem::exists(resultsDir)) {
            std::filesystem::create_directory(resultsDir);
            std::cout << "\nCreated directory: " << resultsDir << std::endl;
        }
        
        std::string csvFilename = resultsDir + "/mpi_benchmark_results.csv";
        bool fileExists = std::filesystem::exists(csvFilename);
        
        std::ofstream csvFile(csvFilename, std::ios::app);
        
        if (!csvFile.is_open()) {
            std::cerr << "WARNING: Could not open " << csvFilename << " for writing." << std::endl;
        } else {
            // Write header if file is new
            if (!fileExists) {
                csvFile << "Timestamp,Rank_Count,Execution_Time_ms,Total_Energy" << std::endl;
            }
            
            // Get current timestamp
            auto now = std::chrono::system_clock::now();
            auto now_time_t = std::chrono::system_clock::to_time_t(now);
            std::tm now_tm;
            localtime_s(&now_tm, &now_time_t);
            
            std::ostringstream timestamp;
            timestamp << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S");
            
            // Write benchmark data
            csvFile << timestamp.str() << ","
                    << size << ","
                    << std::fixed << std::setprecision(2) << executionTime << ","
                    << totalEnergy << std::endl;
            
            csvFile.close();
            std::cout << "Benchmark results appended to: " << csvFilename << std::endl;
        }
    }
    
    MPI_Finalize();
    return 0;
}