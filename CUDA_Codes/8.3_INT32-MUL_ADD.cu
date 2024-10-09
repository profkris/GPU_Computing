#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel to perform both INT32 multiplication and addition, and print SM and warp IDs
__global__ void int32Kernel(int *a, int *b, int *c, int *d, int n) {
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int smId;
    int warpId;

    // Using asm intrinsic to get the SM ID
    asm("mov.u32 %0, %%smid;" : "=r"(smId));
    // Calculate the warp ID
    warpId = globalThreadId / warpSize;

    if (globalThreadId < n) {
        // Perform multiplication
        int mul_result = a[globalThreadId] * b[globalThreadId];
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

    const int N = 128*40*100000;  // Total number of elements for INT32
    const int size_int32 = N * sizeof(int);
    
    // Print sizes of INT32 variables
    printf("Size of int (INT32): %zu bytes\n", sizeof(int));

    // Host memory allocation
    int *h_a_int32, *h_b_int32, *h_c_int32, *h_d_int32;
    h_a_int32 = (int*)malloc(size_int32);
    h_b_int32 = (int*)malloc(size_int32);
    h_c_int32 = (int*)malloc(size_int32);
    h_d_int32 = (int*)malloc(size_int32);

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a_int32[i] = i;
        h_b_int32[i] = 2 * i;
        h_c_int32[i] = 3 * i;
    }

    // Device memory allocation
    int *d_a_int32, *d_b_int32, *d_c_int32, *d_d_int32;
    cudaMalloc((void**)&d_a_int32, size_int32);
    cudaMalloc((void**)&d_b_int32, size_int32);
    cudaMalloc((void**)&d_c_int32, size_int32);
    cudaMalloc((void**)&d_d_int32, size_int32);

    // Copy data from host to device
    cudaMemcpy(d_a_int32, h_a_int32, size_int32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_int32, h_b_int32, size_int32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c_int32, h_c_int32, size_int32, cudaMemcpyHostToDevice);

    // Time measurement for INT32 operations
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    int32Kernel<<<1, 128>>>(d_a_int32, d_b_int32, d_c_int32, d_d_int32, N); // Combined INT32 operations
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("INT32 Multiplication and Addition Time: %.2f ms\n", elapsedTime);

    // Copy results back to host
    cudaMemcpy(h_d_int32, d_d_int32, size_int32, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a_int32);
    cudaFree(d_b_int32);
    cudaFree(d_c_int32);
    cudaFree(d_d_int32);

    // Free host memory
    free(h_a_int32);
    free(h_b_int32);
    free(h_c_int32);
    free(h_d_int32);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
