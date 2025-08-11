#include <stdint.h>
#include <stdlib.h>
#include "common.h"


uint32_t *shuffle_indices(uint32_t num_samples, uint8_t seed) {
    uint32_t *indices = malloc(num_samples * sizeof(uint32_t));
    for (uint32_t i = 0; i < num_samples; ++i)
        indices[i] = i;
    
    srand(seed);
    for (uint32_t init_index = 0; init_index < num_samples; ++init_index) {
        uint32_t index_to_swap = rand() % num_samples;
        uint32_t temp = indices[init_index];
        indices[init_index] = indices[index_to_swap];
        indices[index_to_swap] = temp;
    }
    return indices;
}