#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

__global__ void hello() {

    int threadID = (gridDim.x * blockDim.x * threadIdx.y)+(blockDim.x * blockIdx.x)+threadIdx.x;
    printf("Global TID : %d |I am thread (X : %d, Y: %d, Z: %d) of block (X: %d, Y: %d, Z: %d) in the grid\n",
           threadID,threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z );

}


void printDims(dim3 gridDim, dim3 blockDim) {
    printf("Grid Dimensions : {%d, %d, %d} blocks. \n",
    gridDim.x, gridDim.y, gridDim.z);

    printf("Block Dimensions : {%d, %d, %d} threads.\n",
    blockDim.x, blockDim.y, blockDim.z);
}

int main(int argc, char **argv) {


    dim3 gridDim(2);     // 2 blocks in x direction, y, z default to 1
    dim3 blockDim(2,2);  // 4 threads per block: 2 in x direction, 2 in y

    printDims(gridDim, blockDim);

    printf("From each thread:\n");
    hello<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();      // need for printfs in kernel to flush

    return 0;
}
