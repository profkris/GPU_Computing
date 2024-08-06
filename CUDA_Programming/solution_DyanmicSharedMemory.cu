#include <stdio.h>
#include <cuda_runtime.h>

__global__ void testDynamicSharedMemory(int *blockSums) {
    extern __shared__ int dynamicSharedMemory[];
    int tid = threadIdx.x;
    dynamicSharedMemory[tid] = tid;

    __syncthreads();

    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < blockDim.x; ++i) {
            sum += dynamicSharedMemory[i];
        }

        // Store the sum of this block in global memory
        blockSums[blockIdx.x] = sum;
    }
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("Total shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);

    size_t maxSharedMemory = prop.sharedMemPerBlock;
    int threadsPerBlock = 256;
    int numBlocks = 4;

    // Allocate memory to store the sums of each block
    int *d_blockSums, *h_blockSums;
    h_blockSums = (int*)malloc(numBlocks * sizeof(int));
    cudaMalloc(&d_blockSums, numBlocks * sizeof(int));

    // Launch the kernel with the maximum dynamic shared memory allocation
    testDynamicSharedMemory<<<numBlocks, threadsPerBlock, maxSharedMemory>>>(d_blockSums);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    // Copy the block sums back to the host
    cudaMemcpy(h_blockSums, d_blockSums, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    // Aggregate the results from all blocks on the host
    int totalSum = 0;
    for (int i = 0; i < numBlocks; ++i) {
        totalSum += h_blockSums[i];
    }
    printf("Total Sum: %d\n", totalSum);

    // Free memory
    free(h_blockSums);
    cudaFree(d_blockSums);

    return 0;
}
