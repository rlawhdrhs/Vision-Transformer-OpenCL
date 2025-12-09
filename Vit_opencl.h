#include "Network.h"


#ifndef _ViT_opencl_H
#define _ViT_opencl_H

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

#include <CL/cl.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

void ViT_opencl(ImageData *image, Network *networks, float **prb);
void benchmark_tile_sizes(ImageData *image, Network *networks);
char *get_source_code(const char *file_name, size_t *len);

extern int g_tile_size;  // Runtime tile size variable

#endif
#pragma once