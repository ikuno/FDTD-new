#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "Cuda_wapper.h"

/* __global__ void GPUKernel(GLubyte *tex_data){ */
/*   unsigned int index; */
/*   index = blockDim.x * blockIdx.x + threadIdx.x; */
/*   tex_data[index*3+0] = (GLubyte)index%256; */
/*   tex_data[index*3+1] = (GLubyte)(1000-index)%256; */
/*   tex_data[index*3+2] = (GLubyte)(1000-index)%125; */
/* } */
/*  */
/* void LaunchGPUKernel(GLubyte *tex_data){ */
/*   GPUKernel<<<100, 100>>>(tex_data); */
/* } */

__global__ void GPU_E(float *ez, float *hx, float *hy, float *cezx, float *cezxl, float *cezy, float *cezyl, float *ezx, float *ezy, float *cez, float *cezlx, float *cezly, float step, int t, int L, int width, int height, int power_x, int power_y){
  int i, j, index;
  float pulse;
  j = blockDim.x * blockIdx.x + threadIdx.x;
  i = blockDim.y * blockIdx.y + threadIdx.y;

  index = width * j * i;

  if( i>= height-1 || (j>=width-1) || i==0 || j==0){
    return;
  }

  pulse = sin(((( t - 1 )%(int)step)+1)*2.0*M_PI/step);

  //Ez
  if(i==power_x && j==power_y){
    ez[index] = 1.0/376.7 * pulse;
  }else{
    ez[index] = cez[index] * ez[index] + cezlx[index] * (hy[index]-hy[index-1]) - cezly[index] * (hx[index-width]);
  }

  if(i<L || (i>height-L-1) || j<L || (j>width-L-1)){
    ezx[index] = cezx[index] * ezx[index] + cezxl[index] * (hy[index] - hy[index-1]);
    ezy[index] = cezy[index] * ezy[index] - cezyl[index] * (hx[index] - hx[index-width]);
    ez[index] = ezx[index] + ezy[index];
  }
}

__global__ void GPU_H(float *ez, float *hx, float *hy, float *chyx, float *chyxl, float *chxy, float *chxyl, float *hxy, float *hyx, float *chxly, float *chylx, int L, int width, int height){
  int i, j, index;
  
  j = blockDim.x * blockIdx.x + threadIdx.x;
  i = blockDim.y * blockIdx.y + threadIdx.y;
  index = width * j + i;

  if( i>= height-1 || j>=width-1 ){
    return;
  }

  hx[index] = hx[index] - (chxly[index] * (ez[index+width]-ez[index]));

  if(i<L || (i>height-L-1) || j<L || (j>width-L-1) ){
    hxy[index] = chxy[index] * hxy[index] - chxyl[index] * (ez[index+width] - ez[index]);
    hx[index] = hxy[index];
  }

  if( i==0 || j==0 ){
    return ;
  }

  hy[index] = hy[index] + (chylx[index] * (ez[index+1] - ez[index]));

  if(i<L || (i>height-L-1) || j<L || (j>width-L-1) ){
    hyx[index] = chyx[index] * hyx[index] - chyxl[index] * (ez[index+1] - ez[index]);
    hy[index] = hyx[index];
  }
}

__global__ void GPU_Data(GLubyte *data, float *ez, float yellow, float green, float blue, float max, float min, float width, float height){
  int i, j;
  float v;
  int index;

  j = blockDim.x * blockIdx.x + threadIdx.x;
  i = blockDim.y * blockIdx.y + threadIdx.y;

  index = width*j+i;
  v = ez[index];

  if(v > yellow){
    /* float d = mapping(v, yellow, *max, 0, 255); */
    float d = 0 + (255 - 0) * ((v - yellow)/(max - yellow));
    data[index * 3 + 0] = (GLubyte)255;
    data[index * 3 + 1] = (GLubyte)d;
    data[index * 3 + 2] = (GLubyte)0;
  }else if(v > green){
    data[index * 3 + 0] = (GLubyte)((v-green)/(yellow-green)*255);
    data[index * 3 + 1] = (GLubyte)255;
    data[index * 3 + 2] = (GLubyte)0;
  }else if(v > blue){
    data[index * 3 + 0] = (GLubyte)0;
    data[index * 3 + 1] = (GLubyte)255;
    data[index * 3 + 2] = (GLubyte)(255-(v-blue)/(green-blue)*255);
  }else{
    /* float d = mapping(v, *min, blue, 255, 0); */
    float d = 255 + (0 - 255) * ((v - min)/(blue - min));
    data[index * 3 + 0] = (GLubyte)0;
    data[index * 3 + 1] = (GLubyte)d;
    data[index * 3 + 2] = (GLubyte)255;
  }
}

void GPU_AllocInit(float *ez, float *hx, float *hy,
                  float *cezx, float *cezxl, float *chyx, float *chyxl, float *cezy, float *cezyl, float *chxy, float *chxyl,
                  float *ezx, float *ezy, float *hxy, float *hyx,
                  float *cez, float *cezlx, float *cezly, float *chxly, float *chylx, int gx, int gy){
  int size = gx * gy * sizeof(float);

  cudaMalloc((void**)&ez, size);
  cudaMalloc((void**)&hx, size);
  cudaMalloc((void**)&hy, size);


  cudaMalloc((void**)&cezx, size);
  cudaMalloc((void**)&cezxl, size);
  cudaMalloc((void**)&chyx, size);
  cudaMalloc((void**)&chyxl, size);
  cudaMalloc((void**)&cezy, size);
  cudaMalloc((void**)&cezyl, size);
  cudaMalloc((void**)&chxy, size);
  cudaMalloc((void**)&chxyl, size);

  cudaMalloc((void**)&ezx, size);
  cudaMalloc((void**)&ezy, size);
  cudaMalloc((void**)&hxy, size);
  cudaMalloc((void**)&hyx, size);

  cudaMalloc((void**)&cez, size);
  cudaMalloc((void**)&cezlx, size);
  cudaMalloc((void**)&cezly, size);
  cudaMalloc((void**)&chxly, size);
  cudaMalloc((void**)&chylx, size);
}

void GPU_Free(float *ez, float *hx, float *hy,
                  float *cezx, float *cezxl, float *chyx, float *chyxl, float *cezy, float *cezyl, float *chxy, float *chxyl,
                  float *ezx, float *ezy, float *hxy, float *hyx,
                  float *cez, float *cezlx, float *cezly, float *chxly, float *chylx){
  cudaFree(ez);
  cudaFree(hx);
  cudaFree(hy);

  cudaFree(cezx);
  cudaFree(cezxl);
  cudaFree(chyx);
  cudaFree(chyxl);
  cudaFree(cezy);
  cudaFree(cezyl);
  cudaFree(chxy);
  cudaFree(chxyl);
  

  cudaFree(ezx);
  cudaFree(ezy);
  cudaFree(hxy);
  cudaFree(hyx);


  cudaFree(cez);
  cudaFree(cezlx);
  cudaFree(cezly);
  cudaFree(chxly);
  cudaFree(chylx);
}

void LaunchGPUKernel(GLubyte *data, float *ez, float *hx, float *hy, float *cezx, float *cezxl, float *cezy, float *cezyl, float *ezx, float *ezy, float *cez, float *cezlx, float *cezly, float *chyx, float *chyxl, float *chxy, float *chxyl, float *hxy, float *hyx, float *chxly ,float *chylx, int t, int L, int power_x, int power_y, int width, int height, float yellow, float green, float blue, float max, float min, float step){

  dim3 grid(width / 16+1, height / 16+1);
  dim3 block(16, 16, 1);

  GPU_E <<< grid, block >>> (ez, hx, hy, cezx, cezxl, cezy, cezyl, ezx, ezy, cez, cezlx, cezly, step, t, L, width, height, power_x, power_y);
  GPU_H <<< grid, block >>> (ez, hx, hy, chyx, chyxl, chxy, chxyl, hxy, hyx, chxly, chylx, L, width, height);
  GPU_Data <<<grid, block >>> (data, ez, yellow, green, blue, max, min, width, height);
}

void InitPBO(GLuint *pbo, struct cudaGraphicsResource **pbo_res){
  cudaGraphicsGLRegisterBuffer(pbo_res, *pbo, cudaGraphicsRegisterFlagsWriteDiscard);
}

void CUDA_END(struct cudaGraphicsResource **pbo_res, GLubyte *tex_data){
  cudaGraphicsUnregisterResource(*pbo_res);
  cudaFree(tex_data);
  cudaDeviceReset();
}

void CUDA_kernel_befor(GLubyte *tex_data, GLuint *pbo, GLuint *tex,  struct cudaGraphicsResource **pbo_res){
  cudaGraphicsMapResources(1, pbo_res, 0);
  cudaGraphicsResourceGetMappedPointer((void**)tex_data, NULL, *pbo_res);
}

void CUDA_kernel_after(struct cudaGraphicsResource **pbo_res){
  cudaGraphicsUnmapResources(1, pbo_res, 0);
}
