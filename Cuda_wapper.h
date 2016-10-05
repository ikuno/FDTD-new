#ifndef CUDA_WAPPER_H_INCLUDED__
#define CUDA_WAPPER_H_INCLUDED__

// void LaunchGPUKernel(GLubyte *tex_data);

void LaunchGPUKernel(GLubyte *data, float **ez, float **hx, float **hy, float **cezx, float **cezxl, float **cezy, float **cezyl, float **ezx, float **ezy, float **cez, float **cezlx, float **cezly, float **chyx, float **chyxl, float **chxy, float **chxyl, float **hxy, float **hyx, float **chxly ,float **chylx, int t, int L, int power_x, int power_y, int width, int height, float yellow, float green, float blue, float max, float min, float step);

// void hoge(void);
#endif //CUDA_WAPPER_H_INCLUDED__

