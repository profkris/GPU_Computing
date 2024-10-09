#include <cuda_runtime.h>
#include <stdio.h>

// Kernel function for demonstration (simple example kernel with addition)
__global__ void myKernelAdd(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];  // Simple addition
    }
}

// Kernel function for demonstration (simple example kernel without addition)
__global__ void myKernelMul(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];  // Example operation
    }
}

void checkOccupancy(int threadsPerBlock) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int maxActiveBlocksPerSMAdd;
    int maxActiveBlocksPerSMMul;

    cudaFuncAttributes attr_Mul;
    cudaFuncGetAttributes(&attr_Mul, myKernelMul);
    
    cudaFuncAttributes attr_Add;
    cudaFuncGetAttributes(&attr_Add, myKernelAdd);
    
    
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocksPerSMAdd,
        myKernelAdd,
        threadsPerBlock,
        0 // Assuming no dynamic shared memory is used
    );

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocksPerSMMul,
        myKernelMul,
        threadsPerBlock,
        0 // Assuming no dynamic shared memory is used
    );

    printf("For Kernel with Addition Operation:\n");
    printf("Max Active Blocks per SM: %d\n", maxActiveBlocksPerSMAdd);
    printf("Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Threads per Block: %d\n", threadsPerBlock);
    printf("Register Count per Thread: %d\n", attr_Add.numRegs);
    printf("Shared Memory per Block: %zu bytes\n", attr_Add.sharedSizeBytes);

    printf("\nFor Kernel with Multiplication Operation:\n");
    printf("Max Active Blocks per SM: %d\n", maxActiveBlocksPerSMMul);
    printf("Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Threads per Block: %d\n", threadsPerBlock);
    printf("Register Count per Thread: %d\n", attr_Mul.numRegs);
    printf("Shared Memory per Block: %zu bytes\n", attr_Mul.sharedSizeBytes);
}

int main() {
    checkOccupancy(32);
    return 0;
}
