#include <stdio.h>
#include <cuda_runtime.h>


int main() {
    int device_count;
    cudaGetDeviceCount(&device_count);

    printf("Device count is %d\n", device_count);

    cudaDeviceProp device_prop;
    for (unsigned int i = 0; i < device_count; ++i) {
        cudaGetDeviceProperties(&device_prop, i);

        // Print the device information.
        printf("Device index %d\n", i);
        printf("Max thread per block: %d\n", device_prop.maxThreadsPerBlock);
        printf("Number of SM: %d\n", device_prop.multiProcessorCount);
        printf("Clock frequency: %d\n", device_prop.clockRate);
        
        printf(
            "Number of allowed threads for each dimension x, y, z: %d, %d, %d\n",
            device_prop.maxThreadsDim[0],
            device_prop.maxThreadsDim[1],
            device_prop.maxThreadsDim[2]);
        
        printf(
            "Number of allowed blocks for each dimension x, y, z: %d, %d, %d\n",
            device_prop.maxGridSize[0],
            device_prop.maxGridSize[1],
            device_prop.maxGridSize[2]);
        
        printf("Number of registers available in each SM: %d\n", device_prop.regsPerBlock);
        printf("Size of warps: %d\n", device_prop.warpSize);
    }

    return 0;
}