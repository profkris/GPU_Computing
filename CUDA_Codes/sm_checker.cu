#include <stdio.h>

// Kernel function to record SM IDs
__global__ void recordSmIds(int *smIds, int numSm, int *counter)
{
    // Get the SM ID (using blockIdx.x as a proxy)
    int smId = blockIdx.x;

    // Use atomic operation to record each SM ID only once
    if (atomicCAS(&smIds[smId], -1, smId) == -1) {
        atomicAdd(counter, 1); // Increment counter for recorded SM IDs
    }
}

int main()
{
    int numSm = 40; // Number of SMs
    int *d_smIds, *d_counter;
    int h_counter = 0;

    // Allocate memory on the device to store SM IDs and counter
    cudaMalloc(&d_smIds, numSm * sizeof(int));
    cudaMalloc(&d_counter, sizeof(int));

    // Initialize device memory
    cudaMemset(d_smIds, -1, numSm * sizeof(int));
    cudaMemcpy(d_counter, &h_counter, sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with enough blocks to cover all SMs
    int threadsPerBlock = 1;
    int blocks = numSm;
    recordSmIds<<<blocks, threadsPerBlock>>>(d_smIds, numSm, d_counter);

    // Synchronize to ensure all operations are completed
    cudaDeviceSynchronize();

    // Copy results back to host
    int h_smIds[numSm];
    cudaMemcpy(h_smIds, d_smIds, numSm * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);

    // Print recorded SM IDs
    printf("Recorded SM IDs:\n");
    for (int i = 0; i < numSm; ++i) {
        if (h_smIds[i] != -1) {
            printf("SM ID: %d\n", h_smIds[i]);
        }
    }

    // Print the number of recorded SMs
    printf("Total SMs recorded: %d\n", h_counter);

    // Free device memory
    cudaFree(d_smIds);
    cudaFree(d_counter);

    return 0;
}
