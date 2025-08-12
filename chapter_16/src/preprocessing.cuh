#ifndef PREPROCESSING
#define PREPROCESSING

#include <stdint.h>
#include "data_loader.cuh"


ImageDataset *prepare_dataset(MNISTDataset *dataset);
ImageDataset *normalize_pixels(ImageDataset *dataset);
ImageDataset *add_padding(ImageDataset *dataset, uint8_t num_padding);


#endif