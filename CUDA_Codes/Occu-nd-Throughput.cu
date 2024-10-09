#include <stdio.h>
#include <cuda_runtime.h>

// FP32 Kernel to perform a simple operation (e.g., addition and multiplication)
__global__ void fp32Kernel(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx] + b[idx];  // Example FP32 operation
    }
}

// Function to measure occupancy
void measureOccupancy() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);  // Query for device 0

    int maxActiveBlocks;
    int blockSize = ThreadsPerBlock;  // Number of threads per block

    // Use the CUDA API to compute the maximum number of active blocks per SM
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, fp32Kernel, blockSize, 0);
    float occupancy = (maxActiveBlocks * ThreadsPerBlock) / (float)prop.maxThreadsPerMultiProcessor;

    printf("Device Name: %s\n", prop.name);
    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Achieved Occupancy: %f\n", occupancy);
}

int main() {
    
    const int SMs = 20;
    const int ThreadsPerBlock = 1;
    const int Testing_factor = 1;
    const int n = SMs * ThreadsPerBlock * Testing_factor;  // Number of elements
    
    const int size = n * sizeof(float);

    float *a, *b, *c;                // Host arrays
    float *d_a, *d_b, *d_c;          // Device arrays

    cudaMallocHost(&a, size);
    cudaMallocHost(&b, size);
    cudaMallocHost(&c, size);
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Initialize host arrays
    for (int i = 0; i < n; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    
    int gridSize = (n + ThreadsPerBlock - 1) / ThreadsPerBlock;  // Number of blocks

    // Measure kernel execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    fp32Kernel<<<gridSize, ThreadsPerBlock>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel Execution Time: %f ms\n", milliseconds);

    // Calculate compute throughput (GFLOPS)
    float gflops = (2.0 * n) / (milliseconds * 1e6);
    printf("FP32 Compute Throughput: %f GFLOPS\n", gflops);

    // Calculate memory throughput (GB/s)
    float totalBytes = 3 * size;  // 2 inputs + 1 output
    float memoryThroughput = totalBytes / (milliseconds * 1e6);
    printf("Memory Throughput: %f GB/s\n", memoryThroughput);

    // Measure occupancy
    measureOccupancy();

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Clean up
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
