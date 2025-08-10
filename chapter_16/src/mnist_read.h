#ifndef MNIST_LOAD
#define MNIST_LOAD

#define MAGIC_UNSIGNED_BYTE 0x08
#define MAGIC_IMAGES_DIM 3
#define MAGIC_LABELS_DIM 1
#define MNIST_LABEL_SIZE 10

#include <stdint.h>


typedef struct {
    // The pixels in the image are stored in a row-major format.
    uint8_t *pixel; 
    uint32_t height;
    uint32_t width;
} MNISTImage;


typedef struct {
    MNISTImage *images;
    uint8_t *labels;
    uint32_t num_samples;
} MNISTDataset;


MNISTDataset *load_mnist_dataset(const char *images_file_path, const char *labels_file_path);


#endif