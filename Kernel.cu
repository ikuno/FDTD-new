#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "Cuda_wapper.h"

__global__ void GPUKernel(GLubyte *tex_data){
  unsigned int index;
  index = blockDim.x * blockIdx.x + threadIdx.x;
  tex_data[index*3+0] = (GLubyte)index%256;
  tex_data[index*3+1] = (GLubyte)(1000-index)%256;
  tex_data[index*3+2] = (GLubyte)(1000-index)%125;
}

void LaunchGPUKernel(GLubyte *tex_data){
  GPUKernel<<<100, 100>>>(tex_data);
}
