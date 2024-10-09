#include <stdio.h>
#include <cuda_runtime.h>

__global__ void countBlocksPerSM(int *blockCountPerSM) {
    int smId;
    asm("mov.u32 %0, %%smid;" : "=r"(smId));

    // Increment the count for the SM ID
    atomicAdd(&blockCountPerSM[smId], 1);

    //printf("Block ID %d executed on SM %d\n", blockIdx.x, smId);
}

int main() {
    const int blocks = 680;  // 40*1*17 blocks
    const int threadsPerBlock = 1;
    const int numSMs = 40;

    // Allocate memory to track the number of blocks per SM
    int *d_blockCountPerSM;
    int h_blockCountPerSM[numSMs] = {0};

    cudaMalloc((void**)&d_blockCountPerSM, numSMs * sizeof(int));
    cudaMemcpy(d_blockCountPerSM, h_blockCountPerSM, numSMs * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    countBlocksPerSM<<<blocks, threadsPerBlock>>>(d_blockCountPerSM);

    // Copy results back to host
    cudaMemcpy(h_blockCountPerSM, d_blockCountPerSM, numSMs * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the number of blocks executed per SM
    for (int i = 0; i < numSMs; i++) {
        printf("SM %d executed %d blocks\n", i, h_blockCountPerSM[i]);
    }

    cudaFree(d_blockCountPerSM);
    return 0;
}
