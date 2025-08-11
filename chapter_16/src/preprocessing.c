#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "data_loader.h"
#include "common.h"


ImageDataset *prepare_dataset(MNISTDataset *dataset) {
    uint8_t *copy_labels = malloc(dataset->num_samples * sizeof(uint8_t));
    Image *copy_images = malloc(dataset->num_samples * sizeof(Image));
    uint32_t *copy_view_indices = malloc(dataset->num_samples * sizeof(uint32_t));
    for (uint32_t i = 0; i < dataset->num_samples; ++i) {
        copy_labels[i] = dataset->labels[i];
        copy_view_indices[i] = i;
        MNISTImage image = dataset->images[i];
        float *copy_pixels = malloc(image.height * image.width * sizeof(float));
        for (uint32_t row = 0; row < image.height; ++row)
            for (uint32_t col = 0; col < image.width; ++col)
                copy_pixels[row * image.width + col] = (float) image.pixels[row * image.width + col];
        copy_images[i].pixels = copy_pixels;
        copy_images[i].height = image.height;
        copy_images[i].width = image.width;
    }

    ImageDataset *copy_dataset = malloc(sizeof(ImageDataset));
    copy_dataset->num_samples = dataset->num_samples;
    copy_dataset->labels = copy_labels;
    copy_dataset->images = copy_images;
    copy_dataset->view_indices = copy_view_indices;
    return copy_dataset;
}


ImageDataset *normalize_pixels(ImageDataset *dataset) {
    for (uint32_t i = 0; i < dataset->num_samples; ++i) {
        Image image = dataset->images[i];
        float *pixels = image.pixels;
        
        for (uint32_t row = 0; row < image.height; ++row)
            for (uint32_t col = 0; col < image.width; ++col)
                pixels[row * image.width + col] /= MAX_PIXEL_VALUE;
    }

    return dataset;
}


ImageDataset *add_padding(ImageDataset *dataset, uint8_t num_padding) {
    // TODO: only modify the "view" and "stride" to keep the original data.
    for (uint32_t i = 0; i < dataset->num_samples; ++i) {
        Image *image = &(dataset->images[i]);
        uint32_t new_height = image->height + 2 * num_padding;
        uint32_t new_width = image->width + 2 * num_padding;
        float *new_pixels = calloc(new_height * new_width, new_height * new_width * sizeof(float));
        for (uint32_t row = 0; row < image->height; ++row)
            for (uint32_t col = 0; col < image->width; ++col)
                new_pixels[(row + num_padding) * image->width + (col + num_padding)] = image->pixels[row * image->width + col];
        
        free(image->pixels);
        image->pixels = new_pixels;
        image->height = new_height;
        image->width = new_width;
    }

    return dataset;
}