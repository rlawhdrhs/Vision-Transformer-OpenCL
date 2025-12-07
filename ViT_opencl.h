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
char *get_source_code(const char *file_name, size_t * len);
#endif#pragma once