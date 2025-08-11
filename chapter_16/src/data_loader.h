#ifndef MNIST_LOAD
#define MNIST_LOAD

#define MAGIC_UNSIGNED_BYTE 0x08
#define MAGIC_IMAGES_DIM 3
#define MAGIC_LABELS_DIM 1
#define MNIST_LABEL_SIZE 10

#include <stdint.h>


typedef struct {
    // The pixels in the image are stored in a row-major format.
    uint8_t *pixels; 
    uint32_t height;
    uint32_t width;
} MNISTImage;


typedef struct {
    MNISTImage *images;
    uint8_t *labels;
    uint32_t num_samples;
} MNISTDataset;


typedef struct {
    float *pixels;
    uint32_t height;
    uint32_t width;
} Image;


typedef struct {
    Image *images;
    uint8_t *labels;
    uint32_t num_samples;
} ImageDataset;

// IO.
MNISTDataset *load_mnist_dataset(const char *images_file_path, const char *labels_file_path);

// Dataset split.
uint32_t *shuffle_indices(uint32_t num_samples, uint8_t seed);
ImageDataset *split_dataset(ImageDataset *dataset, uint32_t *indices, uint32_t num_samples);


#endif