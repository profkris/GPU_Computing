#include <stdlib.h>
#include <stdio.h>

// Function to approximate the number of warp schedulers
int getWarpSchedulers(int major, int minor, int multiprocessorCount) {
    // These values are for illustration purposes and may not be accurate for all GPUs
    if (major == 7) { // Volta and Turing architectures
        return multiprocessorCount * 4; // Each SM has 4 warp schedulers
    } else if (major == 6) { // Pascal architecture
        return multiprocessorCount * 2; // Each SM has 2 warp schedulers
    } else if (major == 5) { // Maxwell architecture
        return multiprocessorCount * 2; // Each SM has 2 warp schedulers
    } else if (major == 3 || major == 2) { // Kepler and Fermi architectures
        return multiprocessorCount * 2; // Each SM has 2 warp schedulers
    } else {
        return multiprocessorCount; // Default assumption, 1 warp scheduler per SM
    }
}

int main()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0)
    {
        printf("There is no device supporting CUDA\n");
    }
    int dev;
    for (dev = 0; dev < deviceCount; ++dev)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        if (dev == 0)
        {
            if (deviceProp.major < 1)
            {
                printf("There is no device supporting CUDA.\n");
            }
            else if (deviceCount == 1)
            {
                printf("There is 1 device supporting CUDA\n");
            }
            else
            {
                printf("There are %d devices supporting CUDA\n", deviceCount);
            }
        }
        int warpSchedulers = getWarpSchedulers(deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);
        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
        printf("  Major revision number:                         %d\n", deviceProp.major);
        printf("  Minor revision number:                         %d\n", deviceProp.minor);
        printf("  Total amount of global memory:                 %ld bytes\n", deviceProp.totalGlobalMem);
        printf("  Total amount of constant memory:               %ld bytes\n", deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:       %ld bytes\n", deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        printf("  Warp size:                                     %d\n", deviceProp.warpSize);
        printf("  Multiprocessor count:                          %d\n",deviceProp.multiProcessorCount);
        printf("  Number of warp schedulers:                     %d\n", warpSchedulers);
        printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],  deviceProp.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %ld bytes\n", deviceProp.memPitch);
        printf("  Texture alignment:                             %ld bytes\n", deviceProp.textureAlignment);
        printf("  Clock rate:                                    %d kilohertz\n", deviceProp.clockRate);
    }
}
