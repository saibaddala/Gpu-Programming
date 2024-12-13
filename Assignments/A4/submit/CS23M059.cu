#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/count.h>

using namespace std;

__global__ void load(int *dRoundHealth, int *dScore, int H)
{
    unsigned tid = threadIdx.x;

    dRoundHealth[tid] = H;
    dScore[tid] = 0;
}

__global__ void playRound(int roundNo, int *dXcoord, int *dYcoord, int *dHealth, int *dRoundHealth, int *dScore, int N, int T)
{
    if (roundNo % T == 0)
        return;
    unsigned srcTank = blockIdx.x;
    unsigned targetTank = (srcTank + roundNo) % T;
    unsigned actualTargetTank = threadIdx.x;

    __shared__ long long minDist;
    minDist = LLONG_MAX;

    long long dist = -1;
    long long int srcVal = (long long)dXcoord[srcTank] * (N + 1) + dYcoord[srcTank];
    long long int tarVal = (long long)dXcoord[targetTank] * (N + 1) + dYcoord[targetTank];
    long long int actualTarVal = (long long)dXcoord[actualTargetTank] * (N + 1) + dYcoord[actualTargetTank];

    if ((long long)(dYcoord[targetTank] - dYcoord[srcTank]) * (dXcoord[actualTargetTank] - dXcoord[srcTank]) == (long long)(dXcoord[targetTank] - dXcoord[srcTank]) * (dYcoord[actualTargetTank] - dYcoord[srcTank]) && dHealth[actualTargetTank] > 0)
    {
        if (srcTank != targetTank && dHealth[srcTank] > 0 && ((tarVal > srcVal && actualTarVal > srcVal) || (tarVal < srcVal && actualTarVal < srcVal)))
        {
            dist = abs(dXcoord[actualTargetTank] - dXcoord[srcTank]) + abs(dYcoord[actualTargetTank] - dYcoord[srcTank]);
            atomicMin(&minDist, dist);
        }
    }

    __syncthreads();

    if (minDist == dist && dHealth[srcTank] > 0)
    {
        dScore[srcTank] = dScore[srcTank] + 1;
        atomicSub(&dRoundHealth[actualTargetTank], 1);
    }
}

//***********************************************

int main(int argc, char **argv)
{
    // Variable declarations
    int M, N, T, H, *xcoord, *ycoord, *score;

    FILE *inputfilepointer;

    // File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer = fopen(inputfilename, "r");

    if (inputfilepointer == NULL)
    {
        printf("input.txt file failed to open.");
        return 0;
    }

    fscanf(inputfilepointer, "%d", &M);
    fscanf(inputfilepointer, "%d", &N);
    fscanf(inputfilepointer, "%d", &T); // T is number of Tanks
    fscanf(inputfilepointer, "%d", &H); // H is the starting Health point of each Tank

    // Allocate memory on CPU
    xcoord = (int *)malloc(T * sizeof(int)); // X coordinate of each tank
    ycoord = (int *)malloc(T * sizeof(int)); // Y coordinate of each tank
    score = (int *)malloc(T * sizeof(int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for (int i = 0; i < T; i++)
    {
        fscanf(inputfilepointer, "%d", &xcoord[i]);
        fscanf(inputfilepointer, "%d", &ycoord[i]);
    }

    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************
    int *dXcoord, *dYcoord, *dRoundHealth, *dScore, roundNo = 1;

    cudaMalloc(&dXcoord, T * sizeof(int));
    cudaMemcpy(dXcoord, xcoord, T * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&dYcoord, T * sizeof(int));
    cudaMemcpy(dYcoord, ycoord, T * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&dRoundHealth, T * sizeof(int));
    cudaMalloc(&dScore, T * sizeof(int));

    thrust::device_vector<int> dHealth(T, H);

    load<<<1, T>>>(dRoundHealth, dScore, H);

    while ((thrust::count_if(dHealth.begin(), dHealth.end(), thrust::placeholders::_1 > 0)) > 1)
    {
        playRound<<<T, T>>>(roundNo, dXcoord, dYcoord, thrust::raw_pointer_cast(dHealth.data()), dRoundHealth, dScore, N, T);
        cudaMemcpy(thrust::raw_pointer_cast(dHealth.data()), dRoundHealth, T * sizeof(int), cudaMemcpyDeviceToHost);
        roundNo++;
    }

    cudaMemcpy(score, dScore, sizeof(int) * T, cudaMemcpyDeviceToHost);

    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************

    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end - start;

    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char *outputfilename = argv[2];
    char *exectimefilename = argv[3];
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename, "w");

    for (int i = 0; i < T; i++)
    {
        fprintf(outputfilepointer, "%d\n", score[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename, "w");
    fprintf(outputfilepointer, "%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    cudaDeviceSynchronize();
    return 0;
}
