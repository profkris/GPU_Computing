#include <stdio.h>
#include <cuda_runtime.h>

const int N = 128 * 4; // Total number of elements
const int BLOCK_SIZE = 40;     // Number of threads per block
const int THREADS_PER_BLOCK = 128; // Number of threads per block

// CUDA kernel to perform addition and multiplication
__global__ void fp32Kernel(float *a, float *b, float *c, float *d, int n) {
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int smId;
    int blockId = blockIdx.x;

    // Using asm intrinsic to get the SM ID
    asm("mov.u32 %0, %%smid;" : "=r"(smId));
    
    if (globalThreadId < n) {
        // Perform multiplication
        float mul_result = a[globalThreadId] * b[globalThreadId];
        // Perform addition
        d[globalThreadId] = mul_result + c[globalThreadId];

        // Print Block ID and SM ID
        printf("Block ID %d on SM %d, Thread %d\n", blockId, smId, globalThreadId);
    }
}

void detectGPUs() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA devices detected.\n");
        return;
    }

    printf("Detected %d CUDA device(s):\n", deviceCount);
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        printf("\nDevice %d: %s\n", i, deviceProp.name);
        printf("  Total Global Memory: %.2f GB\n", (float)deviceProp.totalGlobalMem / (1024 * 1024 * 1024));
        printf("  Number of SMs: %d\n", deviceProp.multiProcessorCount);
        printf("  Warp Size: %d\n", deviceProp.warpSize);
    }
}

int main() {
    detectGPUs(); // Detect and print GPU properties

    const int size_fp32 = N * sizeof(float);

    // Print sizes of FP32 variables
    printf("Size of float (FP32): %zu bytes\n", sizeof(float));

    // Host memory allocation
    float *h_a_fp32, *h_b_fp32, *h_c_fp32, *h_d_fp32;
    h_a_fp32 = (float*)malloc(size_fp32);
    h_b_fp32 = (float*)malloc(size_fp32);
    h_c_fp32 = (float*)malloc(size_fp32);
    h_d_fp32 = (float*)malloc(size_fp32);

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a_fp32[i] = (float)i * 1.0f;
        h_b_fp32[i] = (float)i * 2.0f;
        h_c_fp32[i] = (float)i * 3.0f;
    }

    // Device memory allocation
    float *d_a_fp32, *d_b_fp32, *d_c_fp32, *d_d_fp32;
    cudaMalloc((void**)&d_a_fp32, size_fp32);
    cudaMalloc((void**)&d_b_fp32, size_fp32);
    cudaMalloc((void**)&d_c_fp32, size_fp32);
    cudaMalloc((void**)&d_d_fp32, size_fp32);

    // Copy data from host to device
    cudaMemcpy(d_a_fp32, h_a_fp32, size_fp32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_fp32, h_b_fp32, size_fp32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c_fp32, h_c_fp32, size_fp32, cudaMemcpyHostToDevice);

    // Time measurement for FP32 operations
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Kernel launch with 40 threads per block and 1 thread per block
    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    fp32Kernel<<<numBlocks, THREADS_PER_BLOCK>>>(d_a_fp32, d_b_fp32, d_c_fp32, d_d_fp32, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("FP32 Multiplication and Addition Time: %.2f ms\n", elapsedTime);

    // Copy results back to host
    cudaMemcpy(h_d_fp32, d_d_fp32, size_fp32, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a_fp32);
    cudaFree(d_b_fp32);
    cudaFree(d_c_fp32);
    cudaFree(d_d_fp32);

    // Free host memory
    free(h_a_fp32);
    free(h_b_fp32);
    free(h_c_fp32);
    free(h_d_fp32);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

