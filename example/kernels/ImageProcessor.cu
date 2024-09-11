#include "example/kernels/ImageProcessor.cuh"
#include "example/tools/CudaHelper.h"

__global__ void GrayScaleKernel(uchar3* dataIn, uchar3* dataInOut, const unsigned N,
    unsigned w, unsigned h, const float mouseX = -1, const float mouseY = -1)
{
    const auto ix = (threadIdx.x + blockDim.x * blockIdx.x);
    const auto iy = (threadIdx.y + blockDim.y * blockIdx.y);

    if (ix >= w || iy >= h)
    {
        return;
    }

    const auto grayColor = 0.299 * dataIn[ix + w * iy].x + 
                          0.587 * dataIn[ix + w * iy].y + 
                          0.114 * dataIn[ix + w * iy].z;
    dataInOut[ix + w * iy].x = grayColor; 
    dataInOut[ix + w * iy].y = grayColor; 
    dataInOut[ix + w * iy].z = grayColor; 
}

ImageProcessor::ImageProcessor(const unsigned width, const unsigned height, 
        const unsigned pboHandle)
{
    this->width = width;
    this->height = height;
    this->dataSize = width * height;
    cudaGraphicsGLRegisterBuffer(&pboRes, pboHandle, 
        cudaGraphicsMapFlagsNone);

    GET_CUDA_ERROR("cudaGraphicsGLRegisterImage\n");

    cudaMalloc((void **)&imagePrtDev, this->dataSize * sizeof(uchar3));

    blockDim = { 16, 16 };
    gridDim = { cuda::GetDivUp(width, blockDim.x), cuda::GetDivUp(height, blockDim.y) };
}

void ImageProcessor::Process()
{
    // 1. Map
    cudaGraphicsMapResources(1, &pboRes, 0);

    // 2. Get pointer to change using kernel
    uchar3 *imagePtr;
    size_t numBytes;
    cudaGraphicsResourceGetMappedPointer((void **)&imagePtr, &numBytes, pboRes);
    GET_CUDA_ERROR("cudaGraphicsResourceGetMappedPointer\n");

    // 3. OPTIONAL: Copy and original image, if you apply a filter 
    // to an image each call based on some params
    // (It doesn't make sence in this example and will apply the same filter again and again)
    if (!isCopied)
    {
        cudaMemcpy(imagePrtDev, imagePtr, 
            sizeof(uchar3) * width * height, cudaMemcpyDeviceToDevice);
        GET_CUDA_ERROR("cudaMemcpy\n");

        isCopied = true;
    }

    // 3. Run kernel
    GrayScaleKernel<<<gridDim, blockDim>>>((uchar3*)imagePrtDev, imagePtr, 
        this->dataSize, width, height);
    GET_CUDA_ERROR("GrayScaleKernel\n");

    // 4. Unmap
    cudaGraphicsUnmapResources(1, &pboRes, 0);
    GET_CUDA_ERROR("cudaGraphicsUnmapResources\n");
}

ImageProcessor::~ImageProcessor()
{
    cudaGraphicsUnregisterResource(pboRes);
    GET_CUDA_ERROR("cudaGraphicsUnregisterResource\n");

    cudaFree(imagePrtDev);
    GET_CUDA_ERROR("cudaFree\n");
}