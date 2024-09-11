#include "example/kernels/CubeRotator.cuh"
#include "example/tools/CudaHelper.h"

__global__ void RotateVertices(float3* vertices, float angle, int num_vertices) 
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vertices) 
    {
        float3 v = vertices[idx];

        const float rotatedX = v.x * cos(angle) - v.y * sin(angle);
        const float rotatedY = v.x * sin(angle) + v.y * cos(angle);
        v.x = rotatedX;
        v.y = rotatedY;

        vertices[idx] = { v.x, v.y, v.z };
    }
}

CubeRotator::CubeRotator(const unsigned vbo, const unsigned numVertices)
    : numVertices{numVertices}
{
    cudaGraphicsGLRegisterBuffer(&vboRes, vbo, 
        cudaGraphicsMapFlagsNone);
    GET_CUDA_ERROR("cudaGraphicsGLRegisterBuffer\n");
}

void CubeRotator::Process()
{
    // 1. Map
    cudaGraphicsMapResources(1, &vboRes, 0);

    size_t num_bytes = numVertices * 3 * 4; // 8 vertices, 3d, 4 bytes each value
    const float angle = 0.6 * 3.141592f / 180.0f; //  0.6 deg

    // 2. Get pointer to change using kernel
    float3* verticesPointer;
    cudaGraphicsResourceGetMappedPointer((void**)&verticesPointer, &num_bytes, vboRes);

    // 3. Run kernel
    RotateVertices<<<1, numVertices>>>(verticesPointer, angle, numVertices);
    
    // 4. Unmap
    cudaGraphicsUnmapResources(1, &vboRes, 0);
}

CubeRotator::~CubeRotator()
{
    cudaGraphicsUnregisterResource(vboRes);
    GET_CUDA_ERROR("cudaGraphicsUnregisterResource\n");
}