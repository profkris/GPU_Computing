#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024          // Size of the matrix
#define TILESIZE_1 4   // Tile size for rows
#define TILESIZE_2 8   // Tile size for columns

// CUDA Kernel for matrix multiplication with two tile sizes
__global__ void matrixMultiplyWithTwoTiles(float *A, float *B, float *C, int n) {
    // Allocate shared memory for two different tile sizes
    __shared__ float sharedMemA[TILESIZE_1][TILESIZE_2];
    __shared__ float sharedMemB[TILESIZE_2][TILESIZE_1];

    int tx = threadIdx.x; // Thread's column index in the tile
    int ty = threadIdx.y; // Thread's row index in the tile
    int row = blockIdx.y * TILESIZE_1 + ty; // Calculate the row index of the element in the matrix
    int col = blockIdx.x * TILESIZE_2 + tx; // Calculate the column index of the element in the matrix

    float sum = 0.0f;

    // Loop over the tiles in the matrix
    for (int t = 0; t < (n + TILESIZE_2 - 1) / TILESIZE_2; ++t) {
        // Load elements into shared memory
        if (row < n && t * TILESIZE_2 + tx < n)
            sharedMemA[ty][tx] = A[row * n + t * TILESIZE_2 + tx];
        else
            sharedMemA[ty][tx] = 0.0f;

        if (col < n && t * TILESIZE_1 + ty < n)
            sharedMemB[ty][tx] = B[(t * TILESIZE_1 + ty) * n + col];
        else
            sharedMemB[ty][tx] = 0.0f;

        __syncthreads();  // Synchronize to make sure the data is loaded

        // Perform the multiplication using shared memory
        for (int k = 0; k < TILESIZE_2; ++k) {
            sum += sharedMemA[ty][k] * sharedMemB[k][tx];
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
    cudaGetDeviceProperties(&prop, 0);  // Query for device 0

    // Correctly reference the kernel function you want to measure
    int maxActiveBlocksPerSM;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSM, matrixMultiplyWithTwoTiles, threadsPerBlock, 0);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return;
    }

    // Calculate SM-level occupancy
    float smOccupancy = (maxActiveBlocksPerSM * threadsPerBlock) / (float)prop.maxThreadsPerMultiProcessor;
    
    int totalSMs = prop.multiProcessorCount;
    int maxBlocksOnGPU = maxActiveBlocksPerSM * totalSMs;

    // Correctly calculate GPU occupancy
    float gpuOccupancy = (float)numBlocks / maxBlocksOnGPU;
    
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, matrixMultiplyWithTwoTiles);

    // Print the results
    printf("Number of SMs: %d\n", totalSMs);
    printf("Max active blocks per SM: %d\n", maxActiveBlocksPerSM);
    printf("SM-level Occupancy: %.2f%%\n", smOccupancy * 100);
    printf("GPU-level Occupancy: %.2f%%\n", gpuOccupancy * 100);
    printf("Register Count per Thread: %d\n", attr.numRegs);
    printf("Shared Memory per Block: %zu bytes\n", attr.sharedSizeBytes);
}


// Function to initialize matrix with random values
void initializeMatrix(float *matrix, int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = rand() % 100;  // Random value between 0 and 99
    }
}

// Function to print the matrix (for debugging)
void printMatrix(float *matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%0.2f ", matrix[i * n + j]);
        }
        printf("\n");
    }
}

int main() {
    int n = N;

    // Allocate host memory
    float *h_A = (float*)malloc(n * n * sizeof(float));
    float *h_B = (float*)malloc(n * n * sizeof(float));
    float *h_C = (float*)malloc(n * n * sizeof(float));

    // Initialize matrices A and B
    initializeMatrix(h_A, n);
    initializeMatrix(h_B, n);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, n * n * sizeof(float));
    cudaMalloc((void**)&d_B, n * n * sizeof(float));
    cudaMalloc((void**)&d_C, n * n * sizeof(float));

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 dimBlock(TILESIZE_2, TILESIZE_1);
    dim3 dimGrid((n + TILESIZE_2 - 1) / TILESIZE_2, (n + TILESIZE_1 - 1) / TILESIZE_1);

    // Launch the matrix multiplication kernel
    matrixMultiplyWithTwoTiles<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);

    // Measure GPU occupancy
    measureGPUOccupancy(dimBlock.x * dimBlock.y, dimGrid.x * dimGrid.y);

    // Copy result matrix from device to host
    cudaMemcpy(h_C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result matrix (for debugging)
    //printMatrix(h_C, n);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
