#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel to perform both FP64 multiplication and addition, and print SM and warp IDs
__global__ void fp64Kernel(double *a, double *b, double *c, double *d, double n) {
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int smId;
    int warpId;

    // Using asm intrinsic to get the SM ID
    asm("mov.u32 %0, %%smid;" : "=r"(smId));
    // Calculate the warp ID
    warpId = globalThreadId / warpSize;

    if (globalThreadId < n) {
        // Perform multiplication
        double mul_result = a[globalThreadId] * b[globalThreadId];
        // Perform addition
        d[globalThreadId] = mul_result + c[globalThreadId];

        // Print SM ID and warp ID
        printf("Thread %d on SM %d in warp %d\n", globalThreadId, smId, warpId);
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

    const double N = 128*40*90999;  // Total number of elements for FP64
    const double size_fp64 = N * sizeof(double);
    
    // Print sizes of FP64 variables
    printf("Size of double (FP64): %zu bytes\n", sizeof(double));

    // Host memory allocation
    double *h_a_fp64, *h_b_fp64, *h_c_fp64, *h_d_fp64;
    h_a_fp64 = (double*)malloc(size_fp64);
    h_b_fp64 = (double*)malloc(size_fp64);
    h_c_fp64 = (double*)malloc(size_fp64);
    h_d_fp64 = (double*)malloc(size_fp64);

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a_fp64[i] = (double)i * 1.0;
        h_b_fp64[i] = (double)i * 2.0;
        h_c_fp64[i] = (double)i * 3.0;
    }

    // Device memory allocation
    double *d_a_fp64, *d_b_fp64, *d_c_fp64, *d_d_fp64;
    cudaMalloc((void**)&d_a_fp64, size_fp64);
    cudaMalloc((void**)&d_b_fp64, size_fp64);
    cudaMalloc((void**)&d_c_fp64, size_fp64);
    cudaMalloc((void**)&d_d_fp64, size_fp64);

    // Copy data from host to device
    cudaMemcpy(d_a_fp64, h_a_fp64, size_fp64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_fp64, h_b_fp64, size_fp64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c_fp64, h_c_fp64, size_fp64, cudaMemcpyHostToDevice);

    // Time measurement for FP64 operations
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    fp64Kernel<<<40, 128>>>(d_a_fp64, d_b_fp64, d_c_fp64, d_d_fp64, N); // Combined FP64 operations
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("FP64 Multiplication and Addition Time: %.2f ms\n", elapsedTime);

    // Copy results back to host
    cudaMemcpy(h_d_fp64, d_d_fp64, size_fp64, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a_fp64);
    cudaFree(d_b_fp64);
    cudaFree(d_c_fp64);
    cudaFree(d_d_fp64);

    // Free host memory
    free(h_a_fp64);
    free(h_b_fp64);
    free(h_c_fp64);
    free(h_d_fp64);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

