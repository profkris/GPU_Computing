 #include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define MATRIX_SIZE 25000
#define TILE_SIZE 16 // Assuming a square tile size for simplicity

// CUDA kernel for matrix multiplication using shared memory
__global__ void matrixMultiply(float *A, float *B, float *C, int width) {
    // Allocate shared memory for tiles of matrices A and B
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Calculate global row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0;

    // Iterate over tiles
    for (int tileIdx = 0; tileIdx < width / TILE_SIZE; ++tileIdx) {
        // Load tiles into shared memory
        tileA[threadIdx.y][threadIdx.x] = A[row * width + tileIdx * TILE_SIZE + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] = B[(tileIdx * TILE_SIZE + threadIdx.y) * width + col];

        // Synchronize threads to ensure all data is loaded
        __syncthreads();

        // Compute partial sum for the tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // Synchronize threads before loading the next tile
        __syncthreads();
    }

    // Write result to matrix C
    if (row < width && col < width) {
        C[row * width + col] = sum;
    }
}

int main() {
    int width = MATRIX_SIZE;
    size_t size = width * width * sizeof(float);

    // Host matrices and result
    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    // Initialize matrices A and B
    for (int i = 0; i < width * width; ++i) {
        h_A[i] = 1.0; // Replace with your initialization
        h_B[i] = 2.0; // Replace with your initialization
    }

    // Device matrices and result
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (width + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel for matrix multiplication
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, width);
    printf("Grid dimensions: (%d, %d, %d)\n", blocksPerGrid.x, blocksPerGrid.y, blocksPerGrid.z);
    printf("Threads dimensions : (%d, %d, %d)\n",threadsPerBlock.x,threadsPerBlock.y,threadsPerBlock.z);
    // Copy matrix C from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify results (print some elements if needed)
    printf("Sample result: C[0][0] = %f\n", h_C[0]);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
