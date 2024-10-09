#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE_1 4
#define TILE_SIZE_2 8

// Kernel function for matrix multiplication
__global__ void matrixMultiplyWithTwoTiles(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float sharedMemA[TILE_SIZE_1][TILE_SIZE_2];
    __shared__ float sharedMemB[TILE_SIZE_2][TILE_SIZE_1];

    float sum = 0.0f;

    for (int t = 0; t < (n + TILE_SIZE_1 - 1) / TILE_SIZE_1; ++t) {
        // Load elements into shared memory
        if (row < n && t * TILE_SIZE_1 + threadIdx.x < n)
            sharedMemA[threadIdx.y][threadIdx.x] = A[row * n + t * TILE_SIZE_1 + threadIdx.x];
        else
            sharedMemA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < n && t * TILE_SIZE_2 + threadIdx.y < n)
            sharedMemB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE_2 + threadIdx.y) * n + col];
        else
            sharedMemB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();  // Synchronize to make sure the data is loaded

        // Perform the multiplication using shared memory
        for (int k = 0; k < TILE_SIZE_1; ++k) {
            sum += sharedMemA[threadIdx.y][k] * sharedMemB[k][threadIdx.x];
        }

        __syncthreads();  // Synchronize before loading new data
    }

    // Write the result to the global memory
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

// Function to measure GPU occupancy
void measureGPUOccupancy(int threadsPerBlock, int numBlocks) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);  // Query for device 0
    if (err != cudaSuccess) {
        printf("CUDA Error while getting device properties: %s\n", cudaGetErrorString(err));
        return;
    }

    int maxActiveBlocksPerSM;
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSM, matrixMultiplyWithTwoTiles, threadsPerBlock, 0);
    if (err != cudaSuccess) {
        printf("CUDA Error while getting max active blocks per SM: %s\n", cudaGetErrorString(err));
        return;
    }

    // Calculate SM-level occupancy
    float smOccupancy = (maxActiveBlocksPerSM * threadsPerBlock) / (float)prop.maxThreadsPerMultiProcessor;

    int totalSMs = prop.multiProcessorCount;
    int maxBlocksOnGPU = maxActiveBlocksPerSM * totalSMs;

    // Calculate the GPU occupancy based on block utilization
    float gpuOccupancy = (float)numBlocks / maxBlocksOnGPU;

    cudaFuncAttributes attr;
    err = cudaFuncGetAttributes(&attr, matrixMultiplyWithTwoTiles);
    if (err != cudaSuccess) {
        printf("CUDA Error while getting function attributes: %s\n", cudaGetErrorString(err));
        return;
    }

    // Print the results
    printf("Number of SMs: %d\n", totalSMs);
    printf("Max active blocks per SM: %d\n", maxActiveBlocksPerSM);
    printf("SM-level Occupancy: %.2f%%\n", smOccupancy * 100);
    printf("GPU-level Occupancy: %.2f%%\n", gpuOccupancy * 100);
    printf("Register Count per Thread: %d\n", attr.numRegs);
    printf("Shared Memory per Block: %zu bytes\n", attr.sharedSizeBytes);
}

int main() {
    int n = 1024; // Matrix size
    int threadsPerBlock = TILE_SIZE_1 * TILE_SIZE_2; // Example value
    int numBlocks = (n + TILE_SIZE_1 - 1) / TILE_SIZE_1 * (n + TILE_SIZE_2 - 1) / TILE_SIZE_2;

    // Allocate and initialize device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n * n * sizeof(float));
    cudaMalloc(&d_B, n * n * sizeof(float));
    cudaMalloc(&d_C, n * n * sizeof(float));

    // Kernel launch
    dim3 blockSize(TILE_SIZE_1, TILE_SIZE_2);
    dim3 gridSize((n + TILE_SIZE_1 - 1) / TILE_SIZE_1, (n + TILE_SIZE_2 - 1) / TILE_SIZE_2);
    matrixMultiplyWithTwoTiles<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error during kernel launch: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error after kernel execution: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Measure GPU occupancy
    measureGPUOccupancy(threadsPerBlock, numBlocks);

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
