#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024  // Matrix size
#define TILE_SIZE 32  // Tile size

// Kernel for matrix multiplication (high register usage)
__global__ void highRegisterUsageKernel(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Multiple local variables to increase register usage
    float sum = 0.0f;
    float temp1, temp2, temp3, temp4, temp5;
    
    if (row < n && col < n) {
        for (int k = 0; k < n; ++k) {
            // Performing additional computations to increase register usage
            temp1 = A[row * n + k] * B[k * n + col];
            temp2 = A[row * n + k] + B[k * n + col];
            temp3 = temp1 + temp2;
            temp4 = temp3 * 2.0f;
            temp5 = temp4 - temp2;
            sum += temp5;
        }
        C[row * n + col] = sum;
    }
}

// Function to check SM-level occupancy
void checkOccupancy(int threadsPerBlock) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int maxActiveBlocksPerSM;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSM, highRegisterUsageKernel, threadsPerBlock, 0);

    printf("Max Active Blocks per SM: %d\n", maxActiveBlocksPerSM);
    printf("Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Threads per Block: %d\n", threadsPerBlock);
}

// Function to measure GPU occupancy
void measureGPUOccupancy(int threadsPerBlock, int numBlocks) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);  // Query for device 0

    int maxActiveBlocksPerSM;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSM, highRegisterUsageKernel, threadsPerBlock, 0);

    // Calculate SM-level occupancy
    float smOccupancy = (maxActiveBlocksPerSM * threadsPerBlock) / (float)prop.maxThreadsPerMultiProcessor;
    
    // Total active blocks on the entire GPU
    int totalActiveBlocks = numBlocks;
    int totalSMs = prop.multiProcessorCount;
    int maxBlocksOnGPU = maxActiveBlocksPerSM * totalSMs;

    // Calculate the GPU occupancy based on block utilization
    float gpuOccupancy = (float)totalActiveBlocks / maxBlocksOnGPU;
    
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, highRegisterUsageKernel);

    // Print the results
    printf("Number of SMs: %d\n", totalSMs);
    printf("Max active blocks per SM: %d\n", maxActiveBlocksPerSM);
    printf("SM-level Occupancy: %.2f%%\n", smOccupancy * 100);
    printf("GPU-level Occupancy: %.2f%%\n", gpuOccupancy * 100);
    printf("Register Count per Thread: %d\n", attr.numRegs);
    printf("Shared Memory per Block: %zu bytes\n", attr.sharedSizeBytes);
}

int main() {
    int threadsPerBlock = 32;  // Number of threads per block
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;  // Number of blocks to cover the entire matrix

    // Check occupancy
    checkOccupancy(threadsPerBlock);

    // Measure GPU occupancy
    measureGPUOccupancy(threadsPerBlock, numBlocks);

    return 0;
}
