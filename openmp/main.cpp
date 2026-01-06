#include <mpi.h>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <iostream>

// ------------------------------------------------------------
// --- PREPROCESS (Öğrencilerin genişleteceği şablon fonksiyonlar)
// --- OpenCV hazır fonksiyonlarını kullanmak yerine öğrenciler cv::Mat kullanarak
// --- kendi tercih ettikleri algoritmaları yazacaklar!!!
// ------------------------------------------------------------
cv::Mat preprocess(const cv::Mat& input) {
    cv::Mat gray, blurred, normalized;

    // 1) Renkten griye
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);

    // 2) Normalizasyon (0-1 float)
    gray.convertTo(normalized, CV_32F, 1.0 / 255.0);

    // 3) Gürültü azaltma (Gaussian blur örnek)
    cv::GaussianBlur(normalized, blurred, cv::Size(3, 3), 0);

    return blurred;  // float görüntü
}

// ------------------------------------------------------------
// --- PROCESS (Mutlaka matris çarpımı içeren bölüm)
// ------------------------------------------------------------

// Örnek: NxN kernel ile konvolüsyon = matris çarpımı yapısı
cv::Mat process_stage(const cv::Mat& img) {
    cv::Mat output = img.clone();

    return output;
}

// ------------------------------------------------------------
// --- POSTPROCESS (Eşikleme, çizgi çıkarma, kaydetme)
// ------------------------------------------------------------
cv::Mat postprocess(const cv::Mat& img) {
    cv::Mat out8u, thresh;

    // Float → 8-bit
    img.convertTo(out8u, CV_8U, 255.0);

    // Basit threshold
    cv::threshold(out8u, thresh, 150, 255, cv::THRESH_BINARY);

    return thresh;
}

// ------------------------------------------------------------
// --- MAIN
// ------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
        std::cout << "MPI started with " << size << " processes\n";

    // Load image only on rank 0
    cv::Mat img;
    if (rank == 0) {
        img = cv::imread("geo-dist.png");
        if (img.empty()) {
            std::cerr << "Rank 0: Failed to load image!\n";
            MPI_Finalize();
            return -1;
        }
    }

    // ------------------------------------------------------------
    // --- ÖĞRENCİLER BURADA MPI SCATTER / BROADCAST YAPACAK ---
    // ------------------------------------------------------------
    // Şimdilik örnek olarak tüm rank’lara aynı görüntüyü gönderelim
    MPI_Bcast(&img.rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&img.cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) img = cv::Mat(img.rows, img.cols, CV_8UC3);
    MPI_Bcast(img.data, img.rows * img.cols * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // ------------------------------------------------------------
    // --- PREPROCESS
    // ------------------------------------------------------------
    cv::Mat pre = preprocess(img);

    // ------------------------------------------------------------
    // --- PROCESS
    // ------------------------------------------------------------
    cv::Mat proc = process_stage(pre);

    // ------------------------------------------------------------
    // --- POSTPROCESS
    // ------------------------------------------------------------
    cv::Mat post = postprocess(proc);

    // Yalnızca rank 0 sonucu kaydetsin
    if (rank == 0) {
        cv::imwrite("geo-dist-output.png", post);
        std::cout << "Output saved.\n";
    }

    MPI_Finalize();


    // OpenMP - run a parallel loop
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int total = omp_get_num_threads();
        printf("Thread %d/%d working\n", id, total);
    }

    return 0;
}
