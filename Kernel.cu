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

__global__ void GPU_E(float **ez, float **hx, float **hy, float **cezx, float **cezxl, float **cezy, float **cezyl, float **ezx, float **ezy, float **cez, float **cezlx, float **cezly, float step, int t, int L, int width, int height, int power_x, int power_y){
  int i, j;
  float pulse;
  j = blockDim.x * blockIdx.x + threadIdx.x;
  i = blockDim.y * blockIdx.y + threadIdx.y;

  if( i>= height-1 || (j>=width-1) || i==0 || j==0){
    return;
  }

  pulse = sin(((( t - 1 )%(int)step)+1)*2.0*M_PI/step);

  //Ez
  if(i==power_x && j==power_y){
    ez[i][j] = 1.0/376.7 * pulse;
  }else{
    ez[i][j] = cez[i][j] * ez[i][j] + cezlx[i][j] * (hy[i][j]-hy[i-1][j]) - cezly[i][j] * (hx[i][j-1]);
  }

  if(i<L || (i>height-L-1) || j<L || (j>width-L-1)){
    ezx[i][j] = cezx[i][j] * ezx[i][j] + cezxl[i][j] * (hy[i][j] - hy[i-1][j]);
    ezy[i][j] = cezy[i][j] * ezy[i][j] - cezyl[i][j] * (hx[i][j] - hx[i][j-1]);
    ez[i][j] = ezx[i][j] + ezy[i][j];
  }
}

__global__ void GPU_H(float **ez, float **hx, float **hy, float **chyx, float **chyxl, float **chxy, float **chxyl, float **hxy, float **hyx, float **chxly, float **chylx, int L, int width, int height){
  int i, j;
  
  j = blockDim.x * blockIdx.x + threadIdx.x;
  i = blockDim.y * blockIdx.y + threadIdx.y;

  if( i>= height-1 || j>=width-1 ){
    return;
  }

  hx[i][j] = hx[i][j] - (chxly[i][j] * (ez[i][j+1]-ez[i][j]));

  if(i<L || (i>height-L-1) || j<L || (j>width-L-1) ){
    hxy[i][j] = chxy[i][j] * hxy[i][j] - chxyl[i][j] * (ez[i][j+1] - ez[i][j]);
    hx[i][j] = hxy[i][j];
  }

  if( i==0 || j==0 ){
    return ;
  }

  hy[i][j] = hy[i][j] + (chylx[i][j] * (ez[i+1][j] - ez[i][j]));

  if(i<L || (i>height-L-1) || j<L || (j>width-L-1) ){
    hyx[i][j] = chyx[i][j] * hyx[i][j] - chyxl[i][j] * (ez[i+1][j] - ez[i][j]);
    hy[i][j] = hyx[i][j];
  }
}

__global__ void GPU_Data(GLubyte *data, float **ez, float yellow, float green, float blue, float max, float min, float width, float height){
  int i, j;
  float v;
  int index;

  j = blockDim.x * blockIdx.x + threadIdx.x;
  i = blockDim.y * blockIdx.y + threadIdx.y;

  v = ez[i][j];
  index = width*j+i;

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

void GPU_AllocInit(int gx, int gy){

}

void LaunchGPUKernel(GLubyte *data, float **ez, float **hx, float **hy, float **cezx, float **cezxl, float **cezy, float **cezyl, float **ezx, float **ezy, float **cez, float **cezlx, float **cezly, float **chyx, float **chyxl, float **chxy, float **chxyl, float **hxy, float **hyx, float **chxly ,float **chylx, int t, int L, int power_x, int power_y, int width, int height, float yellow, float green, float blue, float max, float min, float step){

  dim3 grid(width / 16+1, height / 16+1);
  dim3 block(16, 16, 1);

  GPU_E <<< grid, block >>> (ez, hx, hy, cezx, cezxl, cezy, cezyl, ezx, ezy, cez, cezlx, cezly, step, t, L, width, height, power_x, power_y);
  GPU_H <<< grid, block >>> (ez, hx, hy, chyx, chyxl, chxy, chxyl, hxy, hyx, chxly, chylx, L, width, height);
  GPU_Data <<<grid, block >>> (data, ez, yellow, green, blue, max, min, width, height);
}


