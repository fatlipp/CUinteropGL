#pragma once

#include <cuda_gl_interop.h>

/// @brief Simple Grayscale Processor
class ImageProcessor
{
public:
    ImageProcessor(const unsigned width, const unsigned height, const unsigned pboHandle);
    ~ImageProcessor();

public:
    void Process();

private:
    cudaGraphicsResource *pboRes;
    void* imagePrtDev;
    bool isCopied;

    dim3 blockDim;
    dim3 gridDim;
    unsigned dataSize;
    unsigned width;
    unsigned height;
};