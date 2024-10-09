#include <stdio.h>
#include <cuda_runtime.h>

const int THREADS_PER_BLOCK = 1; // Number of threads per block
const int SMs = 40;
const int Test_Factor = 18;
const int N = THREADS_PER_BLOCK*SMs*Test_Factor; // Number of elements
const int TOTAL_BLOCKS = (N+(THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK); // Total number of blocks to fit on a single SM

// CUDA kernel to perform operations
__global__ void fp32Kernel(float *a, float *b, float *c) {
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int smId;
    
    // Get the SM ID using inline assembly
    asm("mov.u32 %0, %%smid;" : "=r"(smId));

    if (globalThreadId < N) {
        // Perform addition
        c[globalThreadId] = a[globalThreadId] + b[globalThreadId];
        
        // Print Block ID, Thread ID, and SM ID
        printf("Block ID %d on SM %d, Thread %d\n", blockIdx.x, smId, globalThreadId);
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
        printf("  Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Max Threads per SM: %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  Max Blocks per SM: %d\n", deviceProp.maxBlocksPerMultiProcessor);
    }
}

int main() {
    detectGPUs(); // Detect and print GPU properties

    const int size_fp32 = N * sizeof(float);

    // Print sizes of FP32 variables
    printf("Size of float (FP32): %zu bytes\n", sizeof(float));

    // Host memory allocation
    float *h_a_fp32, *h_b_fp32, *h_c_fp32;
    h_a_fp32 = (float*)malloc(size_fp32);
    h_b_fp32 = (float*)malloc(size_fp32);
    h_c_fp32 = (float*)malloc(size_fp32);

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a_fp32[i] = (float)i * 1.0f;
        h_b_fp32[i] = (float)i * 2.0f;
    }

    // Device memory allocation
    float *d_a_fp32, *d_b_fp32, *d_c_fp32;
    cudaMalloc((void**)&d_a_fp32, size_fp32);
    cudaMalloc((void**)&d_b_fp32, size_fp32);
    cudaMalloc((void**)&d_c_fp32, size_fp32);

    // Copy data from host to device
    cudaMemcpy(d_a_fp32, h_a_fp32, size_fp32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_fp32, h_b_fp32, size_fp32, cudaMemcpyHostToDevice);

    // Time measurement
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Kernel launch with specified number of blocks
    fp32Kernel<<<TOTAL_BLOCKS, THREADS_PER_BLOCK>>>(d_a_fp32, d_b_fp32, d_c_fp32);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("FP32 Addition Time: %.2f ms\n", elapsedTime);

    // Copy results back to host
    cudaMemcpy(h_c_fp32, d_c_fp32, size_fp32, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a_fp32);
    cudaFree(d_b_fp32);
    cudaFree(d_c_fp32);

    // Free host memory
    free(h_a_fp32);
    free(h_b_fp32);
    free(h_c_fp32);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
