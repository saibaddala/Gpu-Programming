/**
*   CS6023: GPU Programming 
*   Assignment 1
*   
*   @author: cs22m056
*/

#include <chrono>
#include <fstream>
#include <iostream>
#include <cuda.h>

using std::cin;
using std::cout;


__global__
void CalculateHadamardProduct(long int* A, long int* B, int N) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N * N) {
        
        int row = i / N;
        int col = i % N;

        A[row * N + col] = A[row * N + col] * B[col * N + row];
    }
}

__global__
void FindWeightMatrix(long int* A, long int* B, int N) {

    long int id = blockIdx.x * blockDim.x * blockDim.y 
                + threadIdx.y * blockDim.x
                + threadIdx.x;

    long int row = id / N;
    long int col = id % N;

    if(row < N && col < N) {

        A[row * N + col] = A[row * N + col] > B[row* N + col] ? A[row * N + col] : B[row * N + col];
    }
}

__global__
void CalculateFinalMatrix(long int* A, long int* B, int N) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    int rowA;
    int colA;

    if(row < N && col < N) {

        rowA = row;
        colA = col;
    } else if(row < N && col >= N) {

        rowA = row;
        colA = col - N;
    } else if(row >= N && col < N) {

        rowA = row - N;
        colA = col;
    } else {

        rowA = row - N;
        colA = col - N;
    }

    if(row < 2 * N && col < 2 * N) {

        B[row * 2 * N + col] = A[rowA * N + colA] * B[row * 2 * N + col];
    }
}


int main(int argc, char** argv) {


    int N;
    cin >> N;
    long int* A = new long int[N * N];
    long int* B = new long int[N * N];
    long int* C = new long int[N * N];
    long int* D = new long int[2 * N * 2 * N];


    for (long int i = 0; i < N * N; i++) {
        cin >> A[i];
    }

    for (long int i = 0; i < N * N; i++) {
        cin >> B[i];
    }

    for (long int i = 0; i < N * N; i++) {
        cin >> C[i];
    }

    for (long int i = 0; i < 2 * N * 2 * N; i++) {
        cin >> D[i];
    }


    long int* d_A;
    long int* d_B;
    long int* d_C;
    long int* d_D;

    cudaMalloc((void**) &d_A, N * N * sizeof(long int));
    cudaMalloc((void**) &d_B, N * N * sizeof(long int));

    cudaMemcpy(d_A, A, N * N * sizeof(long int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(long int), cudaMemcpyHostToDevice);


    dim3 threadsPerBlock(1024, 1, 1);
    dim3 blocksPerGrid(ceil(N * N / 1024.0), 1, 1);


    auto start = std::chrono::high_resolution_clock::now();
    CalculateHadamardProduct<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;
    cudaFree(d_B);

    cudaMalloc((void**)&d_C, N * N * sizeof(long int));
    cudaMemcpy(d_C, C, N * N * sizeof(long int), cudaMemcpyHostToDevice);


    threadsPerBlock = dim3(32, 32, 1);
    blocksPerGrid = dim3(ceil(N * N / 1024.0), 1, 1);


    start = std::chrono::high_resolution_clock::now();
    FindWeightMatrix<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, N);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end - start;
    cudaFree(d_C);

    threadsPerBlock = dim3(32, 32, 1);
    blocksPerGrid = dim3(ceil(2 * N / 32.0), ceil(2 * N / 32.0), 1);

    cudaMalloc((void**)&d_D, 2 * N * 2 * N * sizeof(long int));
    cudaMemcpy(d_D, D, 2 * N * 2 * N * sizeof(long int), cudaMemcpyHostToDevice);


    start = std::chrono::high_resolution_clock::now();
    CalculateFinalMatrix<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_D, N);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed3 = end - start;

    // Make sure your final output from the device is stored in d_D.

    /**
     * 
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     * 
    */

    cudaMemcpy(D, d_D, 2 * N * 2 * N * sizeof(long int), cudaMemcpyDeviceToHost);
    cudaFree(d_D);
    cudaFree(d_A);

    std::ofstream file("cuda.out");

    if (file.is_open()) {

        for (int i = 0; i < 2 * N; i++) {

            for (int j = 0; j < 2 * N; j++) {

                file << D[i * 2 * N + j] << " ";
            }
            file << "\n";
        }
        file.close();
    } else {

        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");

    if(file2.is_open()) {

        file2 << elapsed1.count() << "\n";
        file2 << elapsed2.count() << "\n";
        file2 << elapsed3.count() << "\n";
        file2.close();
    } else {

        std::cout << "Unable to open file";
    }

    return 0;
}