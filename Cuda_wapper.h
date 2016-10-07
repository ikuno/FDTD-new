#ifndef CUDA_WAPPER_H_INCLUDED__
#define CUDA_WAPPER_H_INCLUDED__

// void LaunchGPUKernel(GLubyte *tex_data);


void LaunchGPUKernel(GLubyte *data, float *ez, float *hx, float *hy, float *cezx, float *cezxl, float *cezy, float *cezyl, float *ezx, float *ezy, float *cez, float *cezlx, float *cezly, float *chyx, float *chyxl, float *chxy, float *chxyl, float *hxy, float *hyx, float *chxly ,float *chylx, int t, int L, int power_x, int power_y, int width, int height, float yellow, float green, float blue, float max, float min, float step);

void GPU_AllocInit(float *ez, float *hx, float *hy,
                  float *cezx, float *cezxl, float *chyx, float *chyxl, float *cezy, float *cezyl, float *chxy, float *chxyl,
                  float *ezx, float *ezy, float *hxy, float *hyx,
                  float *cez, float *cezlx, float *cezly, float *chxly, float *chylx, int gx, int gy);

void GPU_Free(float *ez, float *hx, float *hy,
                  float *cezx, float *cezxl, float *chyx, float *chyxl, float *cezy, float *cezyl, float *chxy, float *chxyl,
                  float *ezx, float *ezy, float *hxy, float *hyx,
                  float *cez, float *cezlx, float *cezly, float *chxly, float *chylx);


void InitPBO(GLuint *pbo, struct cudaGraphicsResource **pbo_res);

void CUDA_END(struct cudaGraphicsResource **pho_res, GLubyte *tex_data);

void CUDA_kernel_befor(GLubyte *tex_data, GLuint *pbo, GLuint *tex,  struct cudaGraphicsResource **pbo_res);

void CUDA_kernel_after(struct cudaGraphicsResource **pbo_res);

#endif //CUDA_WAPPER_H_INCLUDED__

