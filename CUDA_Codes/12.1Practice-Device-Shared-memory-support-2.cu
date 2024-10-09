#include <stdio.h>
#include <cuda_runtime.h>

__global__ void testDynamicSharedMemory() {
    extern __shared__ int dynamicSharedMemory[];
    int tid = threadIdx.x;
    dynamicSharedMemory[tid] = tid;

    __syncthreads();

    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < blockDim.x; ++i) {
            sum += dynamicSharedMemory[i];
        }
        printf("Sum: %d\n", sum);
    }
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("Total shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);

    size_t maxSharedMemory = prop.sharedMemPerBlock;
    int threadsPerBlock = 256;
    int numBlocks = 4;

    // Launch the kernel with the maximum dynamic shared memory allocation
    testDynamicSharedMemory<<<numBlocks, threadsPerBlock, maxSharedMemory>>>();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    return 0;
}
