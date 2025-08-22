#ifndef COMMON
#define COMMON


#define MAX_PIXEL_VALUE 255
#define LABEL_SIZE 10
#define DATASET_SPLIT_TRAIN_PROPORTION 0.6
#define BATCH_SIZE 256


cudaError_t cudaMallocCheck(void **dev_ptr, size_t size);
void *mallocCheck(size_t size);


#endif