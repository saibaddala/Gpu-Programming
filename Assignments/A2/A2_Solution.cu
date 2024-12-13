/*
we will be given a 2D matrix and a odd size filter we have to perform usual convolution operation
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





__global__
void aks_kernel(long int* d_mat, long int* d_filter, int m,int n,int k,int z,long int* d_ans) 
{
    int row_offset=blockIdx.x;


    //row and colunm on which this thread is working on 
    int row=row_offset*z+threadIdx.y;
    int col=threadIdx.x;

    extern __shared__ long int s[];
    long int *ds_filter=s;
	long int *ds_mat=s+k*k;




    //loading the filter into the shared memory
    if(threadIdx.x==0 && threadIdx.y==0)
        for (int i = 0; i < k; i ++){
            for(int j=0;j<k;j++){
                ds_filter[i*k+j]=d_filter[i*k+j];
            }
        }
  
    __syncthreads();

    /*So this block will be working on z no. of rows so along with that z rows we have to load extra (k-1)/2 rows from the top
     and also an extra (k-1)/2 rows from the bottom 
     sr is the first row and er is the last row to be loaded for this block
    */
    int sr=row_offset*z-(k-1)/2;
    int er=((row_offset+1)*z)-1+(k-1)/2;


    int srr=sr;

    //loading the matrix into the shared memory corresponding z+k rows are loaded into the shared memory
    if(threadIdx.x==0 && threadIdx.y==0){
        for (int r =sr ; r <= er; r++){
            for(int j=0;j<n;j++){
                if(r>=0 && r<m)
                    ds_mat[(r-sr)*n+j]=d_mat[r*n+j];
            }
        }
    }
    __syncthreads();

    long int sum=0;

    //now this thread is working on a particular row and colunm to do the convolution we need the corresponding rows and colunms numbers
    //when filter is placed
    sr=row-(k-1)/2;
    er=row+(k-1)/2;
    int sc=col-(k-1)/2;
    int ec=col+(k-1)/2;
    

    for(int r=sr;r<=er;r++)
        for(int c=sc;c<=ec;c++){
            if(r>=0 && r<m && c>=0 && c<n){
                sum+= ds_filter[(r-sr)*k+c-sc]*ds_mat[(r-srr)*n+c];
            }
        } 

    d_ans[row*n+col]=sum;

}



int main(int argc, char** argv) {

    int m,n,k;
    cin>>m>>n>>k;


    long int* h_mat = new long int[m * n];
    long int* h_ans = new long int[m * n];
    long int* h_filter = new long int[k * k];



    for (long int i = 0; i < m * n; i++) {
        cin>>h_mat[i];
    }

    for (long int i = 0; i < k * k; i++) {
        cin>>h_filter[i];
    }

    /****************************************************Start Here***********************************************************/
    long int* d_mat;
    long int* d_filter;
    long int* d_ans;

    cudaMalloc(&d_mat, m*n * sizeof(long int));
    cudaMemcpy(d_mat,h_mat,m*n * sizeof(long int),cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_ans, m*n * sizeof(long int));


    cudaMalloc(&d_filter, k*k * sizeof(long int));
    cudaMemcpy(d_filter,h_filter,k*k * sizeof(long int),cudaMemcpyHostToDevice);


    
    
    
    
    
    /*Since a block can have maximum 1024 threads so here we are finding maximum no. of rows we can keep in a block ,where n is the number
    of colunms which is number of elements in a row*/
    int x=1024/n;
    //printf("The value of x is %d\n", x);

    /*Since shared memory per block is 48KB and we will be having our filter also in the block so based on the memory limitation
    y variable denotes the maximum no. of rows we can keep in a block*/

    /* (k*k +  (y+k)*n)*8B <=48KB------>k*k is the size of the filter,y is maximum no. of rows along with that we will be having k extra 
    rows (k-1)/2 rows above the first row and (k-1)/2 rows below the bottom row   */
    int y=(6000-k*k)/n-k;
    //printf("The value of y is %d\n", y);

    /*variable z denotes no. of rows we are finally having in a block considering all the situation*/
    int z=min(min(x,y),m);
    //printf("The value of z is %d\n", z);

    

    dim3 threadsPerBlock(n, z, 1);
    int sm_size=k*k+(z+k-1)*n;

    auto start = std::chrono::high_resolution_clock::now();

    aks_kernel<<<ceil((1.0)*m/z),threadsPerBlock,sm_size*sizeof(long int)>>>(d_mat, d_filter, m,n,k,z,d_ans);
    
    
    cudaDeviceSynchronize();
    /********************************************************************************************************************************/
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;
    cudaMemcpy(h_ans, d_ans, m*n* sizeof(long int), cudaMemcpyDeviceToHost);

    
    // Make sure your final output from the device is stored in d_D.

    /**
     * 
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     * 
    */



    
    std::ofstream file("cuda.out");
    if (file.is_open()) {
        for (long int i = 0; i < m; i++) {
            for (long int j = 0; j < n; j++) {
                file << h_ans[i * n + j] << " ";
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
        file2.close();
    } else {
        std::cout << "Unable to open file";
    }

    return 0;
}