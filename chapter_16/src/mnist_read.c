#include <stdio.h>
#include <stdint.h>
#include <byteswap.h>
#include <stdlib.h>

#include "mnist_read.h"


uint8_t *_uint32_to_byte(uint32_t number) {
    uint8_t *array = malloc(4 * sizeof(uint8_t));
    array[0] = (uint8_t)(number >> 24);
    array[1] = (uint8_t)(number >> 16);
    array[2] = (uint8_t)(number >>  8);
    array[3] = (uint8_t)(number >>  0);
    return array;
}


uint32_t _read_uint32_convert_endian(FILE *stream, const uint32_t invalid_return_value, const char *label) {
    uint32_t number;
    if (!fread(&number, sizeof(uint32_t), 1, stream)) {
        printf("Error in reading the %s.\n", label);
        return invalid_return_value;
    }

    return __bswap_32(number);
}


uint8_t _load_dimension_from_idx_file_header(FILE *stream) {
    uint8_t invalid_return_value = 0;
    uint32_t magic_number = _read_uint32_convert_endian(stream, invalid_return_value, "magic number");
    if (magic_number == invalid_return_value)
        return invalid_return_value;

    // Separate the magic number into 4 one-byte components.
    uint8_t *magic_byte_array = _uint32_to_byte(magic_number);

    if (magic_byte_array[2] != MAGIC_UNSIGNED_BYTE) {
        printf("Unsigned byte (0x08) expected for the basic data type, found: %u\n", magic_byte_array[2]);
        return invalid_return_value;
    }

    return magic_byte_array[3];
}


MNISTImage *_load_images_from_idx_file(const char *file_path, uint32_t *num_samples) {
    FILE *stream = fopen(file_path, "rb");

    if (!stream) {
        printf("Error in opening the image file given path %s\n", file_path);
        return NULL;
    }

    uint8_t num_dimensions = _load_dimension_from_idx_file_header(stream);
    if (num_dimensions != MAGIC_IMAGES_DIM) {
        printf("Expected 3 dimensions for image file, found %u\n", num_dimensions);
        return NULL;
    }

    uint32_t invalid_return_value = 0;
    *num_samples = _read_uint32_convert_endian(stream, invalid_return_value, "number of samples");
    uint32_t image_height = _read_uint32_convert_endian(stream, invalid_return_value, "image height");
    uint32_t image_width = _read_uint32_convert_endian(stream, invalid_return_value, "image width");

    if (*num_samples == invalid_return_value || image_height == invalid_return_value || image_width == invalid_return_value)
        return NULL;

    uint32_t image_size = image_height * image_width;
    MNISTImage *images = (MNISTImage *)malloc(*num_samples * sizeof(MNISTImage));
    for (int i = 0; i < *num_samples; ++i) {
        uint8_t *pixels = malloc(image_size * sizeof(uint8_t));
        fread(pixels, image_size * sizeof(uint8_t), 1, stream);

        MNISTImage image = {.pixels=pixels, .height=image_height, .width=image_width};
        images[i] = image;
    }
    fclose(stream);

    return images;
}


uint8_t *_load_labels_from_idx_file(const char *file_path, uint32_t *num_samples) {
    FILE *stream = fopen(file_path, "rb");

    if (!stream) {
        printf("Error in opening the image file given path %s\n", file_path);
        return NULL;
    }

    uint8_t num_dimensions = _load_dimension_from_idx_file_header(stream);
    if (num_dimensions != MAGIC_LABELS_DIM) {
        printf("Expected 1 dimension for label file, found %u\n", num_dimensions);
        return NULL;
    }

    uint32_t invalid_return_value = 0;
    *num_samples = _read_uint32_convert_endian(stream, invalid_return_value, "number of samples");
    if (*num_samples == invalid_return_value)
        return NULL;
    
    uint8_t *labels = malloc(*num_samples * sizeof(uint8_t));
    fread(labels, *num_samples * sizeof(uint8_t), 1, stream);
    fclose(stream);

    return labels;
}


MNISTDataset *load_mnist_dataset(const char *images_file_path, const char *labels_file_path) {
    uint32_t num_images_samples, num_labels_samples;
    MNISTImage *images = _load_images_from_idx_file(images_file_path, &num_images_samples);
    uint8_t *labels = _load_labels_from_idx_file(labels_file_path, &num_labels_samples);

    if (images == NULL || labels == NULL) {
        printf("The images/labels could not be loaded properly.\n");
        return NULL;
    }

    if (num_images_samples != num_labels_samples) {
        printf("The number of images (n=%u) and labels (n=%u) are not consistent.\n", num_images_samples, num_labels_samples);
    }

    MNISTDataset *dataset = malloc(sizeof(MNISTDataset));
    dataset->images = images;
    dataset->labels = labels;
    dataset->num_samples = num_images_samples;
    return dataset;
}
