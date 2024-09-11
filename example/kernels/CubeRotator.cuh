#pragma once

#include <cuda_gl_interop.h>

class CubeRotator
{
public:
    CubeRotator(const unsigned vbo, const unsigned numVertices);
    ~CubeRotator();

public:
    void Process();

private:
    unsigned numVertices;
    cudaGraphicsResource *vboRes;
    dim3 blockDim;
    dim3 gridDim;
};