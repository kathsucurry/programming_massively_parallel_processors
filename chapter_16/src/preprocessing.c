#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "mnist_read.h"


MNISTDataset *normalize_pixels(MNISTDataset *dataset) {
    MNISTDataset *copy = _deep_copy_dataset(dataset);
    copy->num_samples = 35;

    printf("%d\n", dataset->num_samples);
    printf("%d\n", copy->num_samples);

    return copy;

}