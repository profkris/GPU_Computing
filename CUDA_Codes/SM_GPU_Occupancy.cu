#include <stdio.h>
#include <cuda_runtime.h>

// Kernel function for demonstration (simple example kernel)
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

    // Calculate SM-level occupancy
    float smOccupancy = (maxActiveBlocksPerSM * ThreadsPerBlock) / (float)prop.maxThreadsPerMultiProcessor;
    
    // Total active blocks on the entire GPU
    int totalActiveBlocks = numBlocks;
    int totalSMs = prop.multiProcessorCount;
    int maxBlocksOnGPU = maxActiveBlocksPerSM * totalSMs;

    // Calculate the GPU occupancy based on block utilization
    float gpuOccupancy = (float)totalActiveBlocks / maxBlocksOnGPU;
    
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, myKernel);

    // Print the results
    printf("Number of SMs: %d\n", totalSMs);
    printf("Max active blocks per SM: %d\n", maxActiveBlocksPerSM);
    printf("SM-level Occupancy: %.2f%%\n", smOccupancy * 100);
    printf("GPU-level Occupancy: %.2f%%\n", gpuOccupancy * 100);
    printf("Register Count per Thread: %d\n", attr.numRegs);
    printf("Shared Memory per Block: %zu bytes\n", attr.sharedSizeBytes);
}

int main() {
    // Example array size and CUDA kernel launch parameters
    int arraySize = 16*20*1;
    int threadsPerBlock = 16;
    int numBlocks = (arraySize + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate memory on the GPU
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, arraySize * sizeof(float));
    cudaMalloc((void**)&d_b, arraySize * sizeof(float));
    cudaMalloc((void**)&d_c, arraySize * sizeof(float));

    // Launch the kernel
    myKernel<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, arraySize);

    // Measure occupancy
    measureGPUOccupancy(threadsPerBlock, numBlocks);

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
