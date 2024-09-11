#pragma once

namespace cuda
{

inline unsigned GetDivUp(unsigned a, unsigned b) 
{ 
  return (a % b != 0) ? (a / b + 1) : (a / b); 
}

void getLastCudaError(const char *errorMessage, const char *file,
                               const int line);
}

#define GET_CUDA_ERROR(msg) cuda::getLastCudaError(msg, __FILE__, __LINE__);