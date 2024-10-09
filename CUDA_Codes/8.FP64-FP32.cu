#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel to perform both FP64 multiplication and addition
__global__ void fp64Kernel(double *a, double *b, double *c, double *d, int n) {
    int globalThreadId_fp64 = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (globalThreadId_fp64 < n) {
        // Perform multiplication
        double mul_result_fp64 = a[globalThreadId_fp64] * b[globalThreadId_fp64];
        // Perform addition
        d[globalThreadId_fp64] = mul_result_fp64 + c[globalThreadId_fp64];
    }
}

// CUDA kernel to perform both FP32 multiplication and addition
__global__ void fp32Kernel(float *a, float *b, float *c, float *d, int n) {
    int globalThreadId_fp32 = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (globalThreadId_fp32 < n) {
        // Perform multiplication
        float mul_result_fp32 = a[globalThreadId_fp32] * b[globalThreadId_fp32];
        // Perform addition
        d[globalThreadId_fp32] = mul_result_fp32 + c[globalThreadId_fp32];
    }
}

int main() {
    // Total number of elements
    const int N = 1024;  
    const int size_fp64 = N * sizeof(double);
    const int size_fp32 = N * sizeof(float);
    
    // Threads per block and Block Per Grid
    const int ThreadsPerBlock=128;
    const int BlockPerGrid = 1;
    
    // Print sizes of FP64 and FP32 variables
    printf("Size of double (FP64): %zu bytes\n", sizeof(double));
    printf("Size of float (FP32): %zu bytes\n", sizeof(float));
    printf("Size of int: %zu bytes\n", sizeof(int));

    // Host memory allocation
    double *h_a_fp64, *h_b_fp64, *h_c_fp64, *h_d_fp64;
    float *h_a_fp32, *h_b_fp32, *h_c_fp32, *h_d_fp32;

    h_a_fp64 = (double*)malloc(size_fp64);
    h_b_fp64 = (double*)malloc(size_fp64);
    h_c_fp64 = (double*)malloc(size_fp64);
    h_d_fp64 = (double*)malloc(size_fp64);

    h_a_fp32 = (float*)malloc(size_fp32);
    h_b_fp32 = (float*)malloc(size_fp32);
    h_c_fp32 = (float*)malloc(size_fp32);
    h_d_fp32 = (float*)malloc(size_fp32);

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a_fp64[i] = i * 1.0;
        h_b_fp64[i] = i * 2.0;
        h_c_fp64[i] = i * 3.0;

        h_a_fp32[i] = (float)i * 4.0f;
        h_b_fp32[i] = (float)i * 5.0f;
        h_c_fp32[i] = (float)i * 6.0f;
    }

    // Device memory allocation
    double *d_a_fp64, *d_b_fp64, *d_c_fp64, *d_d_fp64;
    float *d_a_fp32, *d_b_fp32, *d_c_fp32, *d_d_fp32;

    cudaMalloc((void**)&d_a_fp64, size_fp64);
    cudaMalloc((void**)&d_b_fp64, size_fp64);
    cudaMalloc((void**)&d_c_fp64, size_fp64);
    cudaMalloc((void**)&d_d_fp64, size_fp64);

    cudaMalloc((void**)&d_a_fp32, size_fp32);
    cudaMalloc((void**)&d_b_fp32, size_fp32);
    cudaMalloc((void**)&d_c_fp32, size_fp32);
    cudaMalloc((void**)&d_d_fp32, size_fp32);

    // Copy data from host to device
    cudaMemcpy(d_a_fp64, h_a_fp64, size_fp64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_fp64, h_b_fp64, size_fp64, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c_fp64, h_c_fp64, size_fp64, cudaMemcpyHostToDevice);

    cudaMemcpy(d_a_fp32, h_a_fp32, size_fp32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_fp32, h_b_fp32, size_fp32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c_fp32, h_c_fp32, size_fp32, cudaMemcpyHostToDevice);

    // Time measurement for FP64 operations
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Combined FP64 operations
    cudaEventRecord(start);
    fp64Kernel<<<BlockPerGrid, ThreadsPerBlock>>>(d_a_fp64, d_b_fp64, d_c_fp64, d_d_fp64, N); 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("FP64 Multiplication and Addition Time: %.2f ms\n", elapsedTime);

    // Time measurement for FP32 operations
    // Combined FP32 operations
    cudaEventRecord(start);
    fp32Kernel<<<BlockPerGrid, ThreadsPerBlock>>>(d_a_fp32, d_b_fp32, d_c_fp32, d_d_fp32, N); 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("FP32 Multiplication and Addition Time: %.2f ms\n", elapsedTime);

    // Copy results back to host
    cudaMemcpy(h_d_fp64, d_d_fp64, size_fp64, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_d_fp32, d_d_fp32, size_fp32, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a_fp64);
    cudaFree(d_b_fp64);
    cudaFree(d_c_fp64);
    cudaFree(d_d_fp64);

    cudaFree(d_a_fp32);
    cudaFree(d_b_fp32);
    cudaFree(d_c_fp32);
    cudaFree(d_d_fp32);

    // Free host memory
    free(h_a_fp64);
    free(h_b_fp64);
    free(h_c_fp64);
    free(h_d_fp64);

    free(h_a_fp32);
    free(h_b_fp32);
    free(h_c_fp32);
    free(h_d_fp32);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
