#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

// ============================================================================
// PREPROCESS STAGE: Convert RGB input to Grayscale
// ============================================================================
void preprocess_stage(const cv::Mat& input, cv::Mat& gray) {
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
}

// ============================================================================
// PROCESS STAGE: Apply 7x7 Sobel Edge Detection (Manual 2D Convolution)
// ============================================================================
void process_stage(const cv::Mat& gray, cv::Mat& output) {
    int rows = gray.rows;
    int cols = gray.cols;
    
    // Initialize output with zeros
    output = cv::Mat::zeros(gray.size(), CV_8UC1);
    
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
    
    // Generic 2D Convolution: Iterate over valid image region (avoiding boundaries)
    // Safe boundary: start at half_kernel (3) to prevent segmentation faults
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
            
            // Clamp to valid pixel range [0, 255]
            output.at<uchar>(i, j) = (magnitude > 255) ? 255 : static_cast<uchar>(magnitude);
        }
    }
}

// ============================================================================
// POSTPROCESS STAGE: Convert to Binary Image using Thresholding
// ============================================================================
void postprocess_stage(const cv::Mat& result, cv::Mat& output) {
    // Apply binary threshold: pixels > 50 become white (255), others black (0)
    cv::threshold(result, output, 50, 255, cv::THRESH_BINARY);
}

// ============================================================================
// MAIN: Serial Pipeline Baseline with 7x7 Sobel Kernel
// ============================================================================
int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "Serial Image Processing Pipeline (7x7 Sobel)" << std::endl;
    std::cout << "==================================================" << std::endl;
    
    // ------------------------------------------------------------------------
    // LOAD INPUT IMAGE
    // ------------------------------------------------------------------------
    std::string imagePath = "input.jpg";
    cv::Mat inputImage = cv::imread(imagePath, cv::IMREAD_COLOR);
    
    if (inputImage.empty()) {
        std::cerr << "ERROR: Could not open or find the image: " << imagePath << std::endl;
        return -1;
    }
    
    std::cout << "Image loaded: " << imagePath << " (" << inputImage.cols << "x" << inputImage.rows << " pixels)" << std::endl;
    
    // ------------------------------------------------------------------------
    // DECLARE PIPELINE VARIABLES
    // ------------------------------------------------------------------------
    cv::Mat grayImage;
    cv::Mat processedImage;
    cv::Mat outputImage;
    
    // ------------------------------------------------------------------------
    // START TIMING (Entire Pipeline)
    // ------------------------------------------------------------------------
    double startTime = static_cast<double>(cv::getTickCount());
    
    // ------------------------------------------------------------------------
    // STAGE 1: PREPROCESS
    // ------------------------------------------------------------------------
    std::cout << "\n[STAGE 1] PREPROCESS: Converting to Grayscale..." << std::endl;
    preprocess_stage(inputImage, grayImage);
    std::cout << "[STAGE 1] PREPROCESS: Completed." << std::endl;
    
    // ------------------------------------------------------------------------
    // STAGE 2: PROCESS (7x7 Sobel Convolution)
    // ------------------------------------------------------------------------
    std::cout << "\n[STAGE 2] PROCESS: Applying 7x7 Sobel Edge Detection (Manual Convolution)..." << std::endl;
    process_stage(grayImage, processedImage);
    std::cout << "[STAGE 2] PROCESS: Completed." << std::endl;
    
    // ------------------------------------------------------------------------
    // STAGE 3: POSTPROCESS (Binary Threshold)
    // ------------------------------------------------------------------------
    std::cout << "\n[STAGE 3] POSTPROCESS: Applying Binary Threshold..." << std::endl;
    postprocess_stage(processedImage, outputImage);
    std::cout << "[STAGE 3] POSTPROCESS: Completed." << std::endl;
    
    // ------------------------------------------------------------------------
    // STOP TIMING
    // ------------------------------------------------------------------------
    double endTime = static_cast<double>(cv::getTickCount());
    double executionTime = (endTime - startTime) * 1000.0 / cv::getTickFrequency();
    
    // ------------------------------------------------------------------------
    // SAVE OUTPUT
    // ------------------------------------------------------------------------
    cv::imwrite("results/output_serial_7x7.png", outputImage);
    
    // ------------------------------------------------------------------------
    // DISPLAY RESULTS
    // ------------------------------------------------------------------------
    std::cout << "\n==================================================" << std::endl;
    std::cout << "Pipeline Execution Time: " << executionTime << " ms" << std::endl;
    std::cout << "Output saved as: results/output_serial_7x7.png" << std::endl;
    std::cout << "==================================================" << std::endl;
    
    return 0;
}
