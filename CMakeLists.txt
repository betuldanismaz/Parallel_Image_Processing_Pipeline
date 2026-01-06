cmake_minimum_required(VERSION 3.16)
project(ImageProcessingProject CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- OpenMP ---
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    message(STATUS "OpenMP FOUND")
endif()

# --- OpenCV ---
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV FOUND: ${OpenCV_VERSION}")
endif()

# --- MPI ---
find_package(MPI REQUIRED)
if(MPI_FOUND)
    message(STATUS "MPI FOUND")
endif()

# --- A) SERIAL VERSİYON ---
add_executable(serial_app serial/main.cpp)
target_link_libraries(serial_app PRIVATE ${OpenCV_LIBS})

# --- B) OPENMP VERSİYON ---
add_executable(openmp_app openmp/main.cpp)
target_link_libraries(openmp_app PRIVATE ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

# --- C) MPI VERSİYON ---
add_executable(mpi_app mpi/main.cpp)
target_link_libraries(mpi_app PRIVATE ${OpenCV_LIBS} MPI::MPI_CXX)


include_directories(${MPI_CXX_INCLUDE_PATH}) 
include_directories(${OpenCV_INCLUDE_DIRS})