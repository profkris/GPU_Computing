#include <stdio.h>
#include <cuda.h>

// Kernel function using both static and dynamic shared memory
__global__ void exampleKernel(float *d_out, const float *d_in1, const float *d_in2, int n) {
    // Static shared memory allocation **Size Declared**
    __shared__ float staticArray[128];

    // Dynamic shared memory allocation **Size Not Declared** but determine at kernel launch time**
    extern __shared__ float dynamicArray[];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) {
        // Load data into static shared memory
        staticArray[threadIdx.x] = d_in1[idx];
        // Load data into dynamic shared memory
        dynamicArray[threadIdx.x] = d_in2[idx];

        __syncthreads();

        // Perform computation using shared memory
        d_out[idx] = staticArray[threadIdx.x] + dynamicArray[threadIdx.x];
    }
}

// Utility function to check for CUDA errors
void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Number of elements
    int n = 256;
    int size = n * sizeof(float);

    // Host arrays
    float h_in1[n], h_in2[n], h_out[n];

    // Initialize host arrays
    for (int i = 0; i < n; i++) {
        h_in1[i] = (float)i;
        h_in2[i] = (float)(2 * i);
    }

    // Device arrays
    float *d_in1, *d_in2, *d_out;

    // Allocate device memory
    checkCudaError(cudaMalloc((void **)&d_in1, size), "Allocating d_in1");
    checkCudaError(cudaMalloc((void **)&d_in2, size), "Allocating d_in2");
    checkCudaError(cudaMalloc((void **)&d_out, size), "Allocating d_out");

    // Copy data from host to device
    checkCudaError(cudaMemcpy(d_in1, h_in1, size, cudaMemcpyHostToDevice), "Copying h_in1 to d_in1");
    checkCudaError(cudaMemcpy(d_in2, h_in2, size, cudaMemcpyHostToDevice), "Copying h_in2 to d_in2");

    // Kernel launch parameters (256+(128-1)/128) =
    int threadsPerBlock = 128;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Calculate shared memory size (dynamic)
    int sharedMemSize = threadsPerBlock * sizeof(float);

    // Launch the kernel
    exampleKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_out, d_in1, d_in2, n);

    // Check for kernel launch errors
    checkCudaError(cudaGetLastError(), "Launching kernel");

    // Copy result from device to host
    checkCudaError(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost), "Copying d_out to h_out");

    // Print results
    for (int i = 0; i < n; i++) {
        printf("Result[%d] = %f\n", i, h_out[i]);
    }

    // Free device memory
    checkCudaError(cudaFree(d_in1), "Freeing d_in1");
    checkCudaError(cudaFree(d_in2), "Freeing d_in2");
    checkCudaError(cudaFree(d_out), "Freeing d_out");

    return 0;
}
