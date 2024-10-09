#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024  // Matrix size
#define TILE_SIZE 6  // Tile size

// Kernel for matrix multiplication (high shared memory usage)
__global__ void highSharedMemoryUsageKernel(float *A, float *B, float *C, int n) {
    // Define a large shared memory array
    __shared__ float sharedMemA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedMemB[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    float sum = 0.0f;

    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load elements into shared memory
        if (row < n && t * TILE_SIZE + tx < n)
            sharedMemA[ty][tx] = A[row * n + t * TILE_SIZE + tx];
        else
            sharedMemA[ty][tx] = 0.0f;

        if (col < n && t * TILE_SIZE + ty < n)
            sharedMemB[ty][tx] = B[(t * TILE_SIZE + ty) * n + col];
        else
            sharedMemB[ty][tx] = 0.0f;

        __syncthreads();  // Synchronize to make sure the data is loaded

        // Perform the multiplication using shared memory
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sharedMemA[ty][k] * sharedMemB[k][tx];
        }

        __syncthreads();  // Synchronize before loading new data
    }

    // Write the result to the global memory
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

// Function to check SM-level occupancy
void checkOccupancy(int threadsPerBlock) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int maxActiveBlocksPerSM;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSM, highSharedMemoryUsageKernel, threadsPerBlock, 0);

    printf("Max Active Blocks per SM: %d\n", maxActiveBlocksPerSM);
    printf("Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Threads per Block: %d\n", threadsPerBlock);
}

// Function to measure GPU occupancy
void measureGPUOccupancy(int threadsPerBlock, int numBlocks) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);  // Query for device 0

    int maxActiveBlocksPerSM;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSM, highSharedMemoryUsageKernel, threadsPerBlock, 0);

    // Calculate SM-level occupancy
    float smOccupancy = (maxActiveBlocksPerSM * threadsPerBlock) / (float)prop.maxThreadsPerMultiProcessor;
    
    // Total active blocks on the entire GPU
    int totalActiveBlocks = numBlocks;
    int totalSMs = prop.multiProcessorCount;
    int maxBlocksOnGPU = maxActiveBlocksPerSM * totalSMs;

    // Calculate the GPU occupancy based on block utilization
    float gpuOccupancy = (float)totalActiveBlocks / maxBlocksOnGPU;
    
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, highSharedMemoryUsageKernel);

    // Print the results
    printf("Number of SMs: %d\n", totalSMs);
    printf("Max active blocks per SM: %d\n", maxActiveBlocksPerSM);
    printf("SM-level Occupancy: %.2f%%\n", smOccupancy * 100);
    printf("GPU-level Occupancy: %.2f%%\n", gpuOccupancy * 100);
    printf("Register Count per Thread: %d\n", attr.numRegs);
    printf("Shared Memory per Block: %zu bytes\n", attr.sharedSizeBytes);
}

int main() {
    int threadsPerBlock = TILE_SIZE * TILE_SIZE;  // Number of threads per block
    int numBlocks = (N + TILE_SIZE - 1) / TILE_SIZE;  // Number of blocks to cover the entire matrix

    // Check occupancy
    checkOccupancy(threadsPerBlock);

    // Measure GPU occupancy
    measureGPUOccupancy(threadsPerBlock, numBlocks);

    return 0;
}
