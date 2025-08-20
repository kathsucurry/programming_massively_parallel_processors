#ifndef MNIST_LOAD
#define MNIST_LOAD

#define MAGIC_UNSIGNED_BYTE 0x08
#define MAGIC_IMAGES_DIM 3
#define MAGIC_LABELS_DIM 1
#define MNIST_LABEL_SIZE 10

#include <stdint.h>

#include "common.h"


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
    uint32_t *view_indices;
} ImageDataset;


void free_MNIST_images(MNISTImage *images, uint32_t num_samples);
void free_MNIST_dataset(MNISTDataset *dataset);

void free_images(Image *images, uint32_t num_samples);
void free_dataset(ImageDataset *dataset);


// IO.
MNISTDataset *load_mnist_dataset(const char *images_file_path, const char *labels_file_path);

// Dataset split.
void shuffle_indices(ImageDataset *dataset, uint8_t seed);
ImageDataset *split_dataset(ImageDataset *dataset, uint32_t begin_index, uint32_t end_index);
void prepare_batch(float X[], uint8_t y[], ImageDataset *dataset, uint32_t num_samples_in_batch);

#endif