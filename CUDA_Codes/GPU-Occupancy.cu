#include <stdio.h>
#include <cuda_runtime.h>

// Kernel function for demonstration
__global__ void myKernel(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx] + b[idx];  // Example operation
    }
}

// Function to measure GPU occupancy
void measureGPUOccupancy(int ThreadsPerBlock, int numBlocks) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);  // Query for device 0

    int maxActiveBlocksPerSM;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSM, myKernel, ThreadsPerBlock, 0);

    float smOccupancy = (maxActiveBlocksPerSM * ThreadsPerBlock) / (float)prop.maxThreadsPerMultiProcessor;
    
    int totalActiveBlocks = numBlocks;
    int totalSMs = prop.multiProcessorCount;
    int maxBlocksOnGPU = maxActiveBlocksPerSM * totalSMs;

    // Calculate the GPU occupancy based on blocks utilization
    float gpuOccupancy = (float)totalActiveBlocks / maxBlocksOnGPU;
    
    printf("Device Name: %s\n", prop.name);
    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Per-SM Occupancy: %f\n", smOccupancy);
    printf("Total Blocks Launched: %d\n", numBlocks);
    printf("Max Blocks Possible on GPU: %d\n", maxBlocksOnGPU);
    printf("Estimated Effective GPU Occupancy: %f\n", gpuOccupancy);
}

int main() {
    const int SMs = 20;
    const int ThreadsPerBlock = 1024;
    const int Testing_factor = 4;
    const int n = SMs * ThreadsPerBlock * Testing_factor;  // Number of elements
    int gridSize = (n + ThreadsPerBlock - 1) / ThreadsPerBlock;  // Number of blocks

    float *a, *b, *c;
    cudaMallocManaged(&a, n * sizeof(float));
    cudaMallocManaged(&b, n * sizeof(float));
    cudaMallocManaged(&c, n * sizeof(float));

    measureGPUOccupancy(ThreadsPerBlock, gridSize);
    
    myKernel<<<gridSize, ThreadsPerBlock>>>(a, b, c, n);
    cudaDeviceSynchronize();

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    return 0;
}
