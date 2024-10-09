#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

__global__ void hello() {

    /*********************************************************************************************/
		int Global_Block_ID =blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;

    /*********************************************************************************************/
    int Threads_Per_Block = blockDim.x * blockDim.y * blockDim.z;

    /*********************************************************************************************/
    int Global_Thread_ID= Global_Block_ID * Threads_Per_Block +((threadIdx.z * blockDim.x * blockDim.y ) + (threadIdx.y *blockDim.x) + threadIdx.x );

    /*********************************************************************************************/

    printf("Global BID : %d| Global TID = %d |I am thread (%d, %d, %d) of block (%d, %d, %d) in the grid\n",
           Global_Block_ID,Global_Thread_ID,threadIdx.x, threadIdx.y, threadIdx.z,
           blockIdx.x, blockIdx.y, blockIdx.z );

}


void printDims(dim3 gridDim, dim3 blockDim) {
    printf("Grid Dimensions : {%d, %d, %d} blocks. \n",
    gridDim.x, gridDim.y, gridDim.z);

    printf("Block Dimensions : {%d, %d, %d} threads.\n",
    blockDim.x, blockDim.y, blockDim.z);
}

int main(int argc, char **argv) {


    dim3 gridDim(2,2);     // 2 blocks in x and y direction, z default to 1
    dim3 blockDim(2,2);  // 4 threads per block: 2 in x direction, 2 in y


    printDims(gridDim, blockDim);

    printf("From each thread:\n");
    hello<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();      // need for printfs in kernel to flush

    return 0;
}
