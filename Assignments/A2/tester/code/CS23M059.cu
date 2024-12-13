/**
 *   CS6023: GPU Programming
 *   Assignment 2
 *   0.0000714
 *   Please don't change any existing code in this file.
 *
 *   Please add necessary memory APIs for your implementation. Use cudaFree()
 *   to free up memory as soon as you're done with an allocation.
 *   This will ensure that you don't run out of memory while running
 *   large test cases. Use the minimum required memory for your
 *   implementation. DO NOT change the kernel configuration parameters.
 */

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>

using namespace std;

using std::cin;
using std::cout;

typedef long long ll;

__constant__ long int filter[2401];

__global__ void convolute(long int *globalMatrix, int totalNums, int offset, int perThread, int threadsReq, long int *ans, int m, int n, int k)
{
    extern __shared__ int matrix[];

    unsigned int t_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (t_id < 1024 * blockIdx.x + threadsReq)
    {
        int loc = 1024 * blockIdx.x + (t_id - 1024 * blockIdx.x) * (perThread);
        int temp = perThread;
        while (temp--)
        {
            if (loc >= 0 && loc < m * n)
                matrix[loc - 1024 * blockIdx.x] = globalMatrix[loc];
            loc++;
        }
    }

    __syncthreads();

    int g_Row = t_id / n;
    int g_Col = t_id % n;
    int start_R = g_Row - k / 2;
    int start_C = g_Col - k / 2;

    int sum = 0;
    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < k; j++)
        {
            int f_ind = i * k + j;
            int r = start_R + i;
            int c = start_C + j;
            int g_ind = r * n + c;
            int val;
            if (r < 0 || r >= m || c < 0 || c >= n)
                val = 0;
            else
            {
                int s_ind = g_ind - 1024 * blockIdx.x;
                if (s_ind >= 0 && s_ind < offset)
                    val = matrix[s_ind];
                else
                    val = globalMatrix[g_ind];
                sum += (val * filter[f_ind]);
            }
        }
    }

    if (g_Row * n + g_Col < m * n)
        ans[g_Row * n + g_Col] = sum;
}

int main(int argc, char **argv)
{

    int m, n, k;
    cin >> m >> n >> k;

    long int *h_mat = new long int[m * n];
    long int *h_filter = new long int[k * k];

    long int *h_ans = new long int[m * n];

    for (long int i = 0; i < m * n; i++)
    {
        cin >> h_mat[i];
    }

    for (long int i = 0; i < k * k; i++)
    {
        cin >> h_filter[i];
    }

    /**
     *
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     *
     **/

    /****************************************************Start Here***********************************************************/
    long int *matrix;
    long int *ans;

    cudaMemcpyToSymbol(filter, h_filter, k * k * sizeof(long int), 0, cudaMemcpyHostToDevice);
    cudaMalloc(&matrix, m * n * sizeof(long int));
    cudaMemcpy(matrix, h_mat, m * n * sizeof(long int), cudaMemcpyHostToDevice);
    cudaMalloc(&ans, m * n * sizeof(long int));

    int blocksReq = ceil((m * n) / 1024.0);
    int computableRows = ceil(1024.0 / n);
    computableRows = computableRows >= m ? m : computableRows;
    int rowsToBestored = computableRows;

    if (computableRows != m && (k / 2) * n < 8 * 1024)
        rowsToBestored += k / 2;
    else if (computableRows != m)
        rowsToBestored += (7 * 1024) / n;

    int size = rowsToBestored * n;
    int offset = rowsToBestored * n;
    int perThread = ceil(size / 1024.0);
    int threadsReq = size / perThread;

    auto start = std::chrono::high_resolution_clock::now(); // keep it just before the kernel launch
    convolute<<<blocksReq, 1024, size * sizeof(int)>>>(matrix, size, offset, perThread, threadsReq, ans, m, n, k);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now(); // keep it just after the kernel launch
    
    cudaFree(matrix);
    cudaMemcpy(h_ans, ans, m * n * sizeof(long int), cudaMemcpyDeviceToHost);
    /*$$$$$$$$$$$$$$$$$$$$$$$$Make sure your final output from the device is stored in h_ans.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/

    std::chrono::duration<double> elapsed1 = end - start;
    /**
     *
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     *
     */

    std::ofstream file("cuda.out");
    if (file.is_open())
    {
        for (long int i = 0; i < m; i++)
        {
            for (long int j = 0; j < n; j++)
            {
                file << h_ans[i * n + j] << " ";
            }
            file << "\n";
        }
        file.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if (file2.is_open())
    {
        file2 << elapsed1.count() << "\n";
        file2.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    return 0;
}
