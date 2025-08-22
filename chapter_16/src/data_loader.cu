#include <stdio.h>
#include <stdint.h>
#include <byteswap.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "data_loader.cuh"
#include "common.cuh"


void free_MNIST_images(MNISTImage *images, uint32_t num_samples) {
    for (uint32_t i = 0; i < num_samples; ++i)
        free(images[i].pixels);
    free(images);
}


void free_MNIST_dataset(MNISTDataset *dataset) {
    free_MNIST_images(dataset->images, dataset->num_samples);
    free(dataset->labels);
    free(dataset);
}


void free_images(Image *images, uint32_t num_samples) {
    if (images == NULL)
        return;
    for (uint32_t i = 0; i < num_samples; ++i)
        free(images[i].pixels);
    free(images);
}

void free_dataset(ImageDataset *dataset) {
    free_images(dataset->images, dataset->num_samples);
    free(dataset->labels);
    free(dataset->view_indices);
    free(dataset);
}


uint8_t *_uint32_to_byte(uint32_t number) {
    uint8_t *array = (uint8_t *)mallocCheck(4 * sizeof(uint8_t));
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

    uint8_t num_dim = magic_byte_array[3];
    free(magic_byte_array);

    return num_dim;
}


MNISTImage *_load_images_from_idx_file(const char *file_path, uint32_t *num_samples) {
    FILE *stream = fopen(file_path, "rb");

    if (!stream) {
        printf("Error in opening the image file given path %s\n", file_path);

        // Get current path.
        char path[200];
        getcwd(path, 200);
        printf("Current working directory: %s\n", path);

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
    MNISTImage *images = (MNISTImage *)mallocCheck(*num_samples * sizeof(MNISTImage));
    for (int i = 0; i < *num_samples; ++i) {
        uint8_t *pixels = (uint8_t *)mallocCheck(image_size * sizeof(uint8_t));
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
    
    uint8_t *labels = (uint8_t *)mallocCheck(*num_samples * sizeof(uint8_t));
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

    MNISTDataset *dataset = (MNISTDataset *)mallocCheck(sizeof(MNISTDataset));
    dataset->images = images;
    dataset->labels = labels;
    dataset->num_samples = num_images_samples;
    return dataset;
}



void shuffle_indices(ImageDataset *dataset, uint8_t seed) {
    srand(seed);
    uint32_t num_samples = dataset->num_samples;
    // printf("In shuffle indices, num_samples is %u\n", num_samples);
    for (uint32_t init_index = 0; init_index < num_samples; ++init_index) {
        uint32_t index_to_swap = rand() % num_samples;
        uint32_t temp = dataset->view_indices[init_index];
        // if (temp > num_samples)
        //     printf("Why is temp > num_samples????\n");
        dataset->view_indices[init_index] = dataset->view_indices[index_to_swap];
        dataset->view_indices[index_to_swap] = temp;
    }
}


/**
 * Allocate dataset, typically used for splitting into training and validation sets.
 * 
 * Note that the images in split_dataset points to the images in the inputted dataset, i.e.,
 * if dataset's memory is freed early, the images in split_datset will also be affected. Setting clear_dataset to true
 * will release the inputted dataset's pointer to the images.
 * 
 */
ImageDataset *split_dataset(ImageDataset *dataset, uint32_t begin_index, uint32_t end_index, bool clear_dataset) {
    uint32_t num_samples = end_index - begin_index;
    if (num_samples > dataset->num_samples) {
        printf("Error in dataset split: number of samples for the new split exceeds the initial number of samples.");
        return NULL;
    }

    ImageDataset *split_dataset = (ImageDataset *)mallocCheck(sizeof(ImageDataset));
    split_dataset->num_samples = num_samples;
    split_dataset->images = (Image *)mallocCheck(num_samples * sizeof(Image));
    split_dataset->labels = (uint8_t *)mallocCheck(num_samples * sizeof(uint8_t));
    split_dataset->view_indices = (uint32_t *)mallocCheck(num_samples * sizeof(uint32_t));
    for (uint32_t i = 0; i < num_samples; ++i) {
        uint32_t old_index = dataset->view_indices[begin_index + i];
        split_dataset->images[i] = dataset->images[old_index];
        split_dataset->labels[i] = dataset->labels[old_index];
        split_dataset->view_indices[i] = i;
    }   

    if (clear_dataset) {
        dataset->images = NULL;
    }
    return split_dataset; 
}


void prepare_batch(
    float X[], uint8_t y[],
    ImageDataset *dataset,
    uint32_t start_index, uint32_t num_samples_in_batch
) {
    uint32_t image_size = dataset->images[0].height * dataset->images[0].width;
    memset(y, 0, num_samples_in_batch * LABEL_SIZE * sizeof(uint8_t));
    for (uint32_t i = 0; i < num_samples_in_batch; ++i) {
        uint32_t index = dataset->view_indices[start_index + i];
        memcpy(&X[i * image_size], dataset->images[index].pixels, image_size * sizeof(float));
        uint8_t label = dataset->labels[index];
        y[i * LABEL_SIZE + label] = 1;
    }
}
