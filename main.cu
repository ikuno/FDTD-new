#include <cassert>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <cstdlib>

// #include <string>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "Program.h"
#include "Camera.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16

#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#define NK_IMPLEMENTATION
#define NK_GLFW_GL3_IMPLEMENTATION
#include "./nuklear/nuklear.h"
#include "./nuklear/nuklear_glfw_gl3.h"

#define MAX_VERTEX_BUFFER 512 * 1024
#define MAX_ELEMENT_BUFFER 128 * 1024

struct nk_context *ctx;

/****************
 *   parameter   *
 ****************/
float c = 2.99792458e8;
float freq = 1.0e9;
float lambda;
float resolution = 20.0;
float delta_x;
float delta_y;
float alpha = 0.5;
float delta_t;
float step;
float mu0;
float sigma0 = 0;
float epsilon0 = 8.854187e-12f;
float epsilonMax;
int M = 4;
int L = 12;
float r0 = -6;
float ecmax;
float Ez_max = 2.060459378159e-03f;
float Ez_min = -7.196258220476e-04f;
float Ez_range;
float Ez_yellow;
float Ez_green;
float Ez_lightblue;

// float pulse;
float T = 0.0;
bool flag = false;
int kt = 1;

float *h_Ez, *h_Hx, *h_Hy, *h_sigma_M, *h_epsilon_M, *h_mu_M;
float *h_ECX, *h_ECY;
float *h_CEZX, *h_CEZXL, *h_CHYX, *h_CHYXL, *h_CEZY, *h_CEZYL, *h_CHXY, *h_CHXYL;
float *h_EZX, *h_EZY, *h_HXY, *h_HYX;
float *h_CEZ, *h_CEZLX, *h_CEZLY, *h_CHXLY, *h_CHYLX;

float *d_Ez, *d_Hx, *d_Hy;
float *d_CEZX, *d_CEZXL, *d_CHYX, *d_CHYXL, *d_CEZY, *d_CEZYL, *d_CHXY, *d_CHXYL;
float *d_EZX, *d_EZY, *d_HXY, *d_HYX;
float *d_CEZ, *d_CEZLX, *d_CEZLY, *d_CHXLY, *d_CHYLX;

int wall_r;
int power_x, power_y;

//camera
float gScrollY = 0.0;





int SIZE_X=1080;
int SIZE_Y=1080;

/* int grid_x=256; */
/* int grid_y=256; */
int GRID_X=1080;
int GRID_Y=1080;

GLFWwindow* gWindow = NULL;
tdogl::Program* gProgram = NULL;
tdogl::Camera gCamera;

GLuint gVAO = 0;
GLuint gVBO = 0;
GLuint gTEX = 0;
GLuint gPBO = 0;
GLubyte *h_g_data;
GLubyte *d_g_data;
struct cudaGraphicsResource *pbo_res;





void free_data(void);

void malloc_Initialdata(void);

void setInitialData(unsigned int width, unsigned int height);

void launchCPUKernel(GLubyte *g_data, float *Ez, float *Hx, float *Hy, float *CEZX, float *CEZXL, float *CHYX, float *CHYXL, float *CEZY, float *CEZYL, float *CHXY, float *CHXYL, float *EZX, float *EZY, float *HXY, float *HYX, float *CEZ, float *CEZLX, float *CEZLY, float *CHXLY, float *CHYLX, float step, unsigned int t, int L, unsigned int width, unsigned int height, float max, float min, float yellow, float green, float lightblue, int power_x, int power_y);

void launchGPUKernel(GLubyte *g_data, float *Ez, float *Hx, float *Hy, float *CEZX, float *CEZXL, float *CHYX, float *CHYXL, float *CEZY, float *CEZYL, float *CHXY, float *CHXYL, float *EZX, float *EZY, float *HXY, float *HYX, float *CEZ, float *CEZLX, float *CEZLY, float *CHXLY, float *CHYLX, float step, unsigned int t, int L, unsigned int width, unsigned int height, float max, float min, float yellow, float green, float lightblue, int power_x, int power_y, int R);

void h_FDTD2d_tm(GLubyte *g_data, float *Ez, float *Hx, float *Hy, float *CEZX, float *CEZXL, float *CHYX, float *CHYXL, float *CEZY, float *CEZYL, float *CHXY, float *CHXYL, float *EZX, float *EZY, float *HXY, float *HYX, float *CEZ, float *CEZLX, float *CEZLY, float *CHXLY, float *CHYLX, float step, unsigned int t, int L, unsigned int width, unsigned int height, float max, float min, float yellow, float green, float lightblue, int power_x, int power_y);

float h_clamp(float x, float a, float b);

__device__ float d_clamp(float x, float a, float b);

__global__ void d_FDTD2d_tm_H(GLubyte *g_data, float *Ez, float *Hx, float *Hy, float *CHYX, float *CHYXL, float *CHXY, float *CHXYL, float *HXY, float *HYX, float *CHXLY, float *CHYLX, int L, unsigned int width, unsigned int height, float max, float min, float yellow, float green, float lightblue);

__global__ void d_FDTD2d_tm_E(float *Ez, float *Hx, float *Hy, float *CEZX, float *CEZXL, float *CEZY, float *CEZYL, float *EZX, float *EZY, float *CEZ, float *CEZLX, float *CEZLY, float step, unsigned int t, int L, unsigned int width, unsigned int height, int power_x, int power_y);

float h_clamp(float x, float a, float b);


void RunGPUKernel(void);
void RunCPUKernel(void);

void InitPBO(GLuint *pbo, unsigned int size, struct cudaGraphicsResource **pbo_res, unsigned int pbo_res_flags);

void InitTexData();

void LoadShaders();

void LoadTriangle();

void Render();

void OnError(int errorCode, const char* msg);

void AppMain();

float CalcFPS(GLFWwindow *gWindow, float theTimeInterval = 1.0, std::string theWindowTitle = "NONE");

void Update(float secondsElapsed);

void CameraInit();

void OnScroll(GLFWwindow *window, double deltaX, double deltaY);

void OnClick(GLFWwindow *window, int button, int action, int mods);

void PEC(GLubyte *h_g_data, float *ez, int X, int Y, int r);

void GUIRender(struct nk_context *ctx, int x, int y);




void GUIRender(struct nk_context *ctx, int x, int y)
{
  struct nk_panel layout;
  struct nk_rect bounds;
  const struct nk_input *in = &ctx->input;
  if (nk_begin(ctx, &layout, "Tools", nk_rect(50, 50, 300, 400),
        NK_WINDOW_BORDER|NK_WINDOW_MOVABLE|NK_WINDOW_SCALABLE|
        NK_WINDOW_MINIMIZABLE|NK_WINDOW_TITLE))
  {
    nk_layout_row_dynamic(ctx, 30, 2);
    bounds = nk_widget_bounds(ctx);
    if(nk_button_label(ctx, "Start/Stop"))
    {
      flag=!flag;
    }
    if(nk_button_label(ctx, "Restart"))
    {
      flag=false;
      kt=1;
      T=0.0;
      setInitialData(x, y);
      CameraInit();
      gCamera.setFieldOfView(50.0);
    }
    nk_layout_row_dynamic(ctx, 20, 2);
    nk_value_float_e(ctx, "Ez_Max", Ez_max);
    nk_value_float_e(ctx, "Ez_Min", Ez_min);

    if (nk_input_is_mouse_hovering_rect(in, bounds))
    {
      const struct nk_style *style;
      struct nk_panel layout;

      style=&ctx->style;

      if(nk_tooltip_begin(ctx, &layout, 130))
      {
        nk_layout_row_dynamic(ctx, style->font.height, 1);
        nk_text(ctx, "K -> MoveUp", 11, NK_TEXT_LEFT);
        nk_layout_row_dynamic(ctx, style->font.height, 1);
        nk_text(ctx, "J -> MoveDown", 13, NK_TEXT_LEFT);
        nk_layout_row_dynamic(ctx, style->font.height, 1);
        nk_text(ctx, "H -> MoveLeft", 13, NK_TEXT_LEFT);
        nk_layout_row_dynamic(ctx, style->font.height, 1);
        nk_text(ctx, "L -> MoveRight", 14, NK_TEXT_LEFT);
        nk_layout_row_dynamic(ctx, style->font.height, 1);
        nk_text(ctx, "Z -> ZoomIn", 11, NK_TEXT_LEFT);
        nk_layout_row_dynamic(ctx, style->font.height, 1);
        nk_text(ctx, "X -> ZoomOut", 12, NK_TEXT_LEFT);
        nk_layout_row_dynamic(ctx, style->font.height, 1);
        nk_text(ctx, "C -> ResetCamera", 16, NK_TEXT_LEFT);
        nk_tooltip_end(ctx);
      }
    }


    /* if (nk_tree_push(ctx, NK_TREE_NODE, "Sampling", NK_MINIMIZED)) */
    /* { */
    /*   int i; */
    /*   static int onoff=nk_false; */
    /*   static int pick=nk_false; */
    /*   nk_layout_row_dynamic(ctx, 20, 4); */
    /*   nk_checkbox_label(ctx, "On/Off", &onoff); */
    /*   nk_selectable_label(ctx, "Pick", NK_TEXT_LEFT, &pick); */
    /*   nk_value_int(ctx, "x", sampling[0]); */
    /*   nk_value_int(ctx, "y", ftoi(GRID_SIZE.y)-sampling[1]); */
    /*   if(onoff==nk_false) */
    /*     sampling_flag=false; */
    /*   else */
    /*     sampling_flag=true; */
    /*  */
    /*   if(pick==nk_false) */
    /*     sampling_pick=false; */
    /*   else */
    /*     sampling_pick=true; */
    /*  */
    /*   nk_layout_row_dynamic(ctx, 60, 1); */
    /*  */
    /*   if(nk_chart_begin(ctx, NK_CHART_LINES, 100, Ez_min, Ez_max)) */
    /*   { */
    /*     for(i=0;i<100;i++) */
    /*     { */
    /*       nk_chart_push(ctx, sampling_list[i]); */
    /*     } */
    /*     nk_chart_end(ctx); */
    /*   } */
    /*  */
    /*   nk_tree_pop(ctx); */
    /* } */

    if(nk_tree_push(ctx, NK_TREE_NODE, "Shape", NK_MINIMIZED))
    {
      nk_layout_row_begin(ctx, NK_STATIC, 20, 3);
      nk_layout_row_push(ctx, 20);
      nk_button_set_behavior(ctx, NK_BUTTON_REPEATER);
      if(nk_button_symbol(ctx, NK_SYMBOL_TRIANGLE_LEFT)){
        wall_r-=1;
        if(wall_r<=0){
          wall_r=0;
        }
      }

      nk_layout_row_push(ctx, 100);
      nk_value_float(ctx, "Width", wall_r);

      nk_layout_row_push(ctx, 20);
      nk_button_set_behavior(ctx, NK_BUTTON_REPEATER);
      if(nk_button_symbol(ctx, NK_SYMBOL_TRIANGLE_RIGHT)){
        wall_r+=1;
      }
      nk_layout_row_end(ctx);
      nk_tree_pop(ctx);
    }

  }
  nk_end(ctx);

}


void OnClick(GLFWwindow *window, int button, int action, int mods)
{
  if(button == GLFW_MOUSE_BUTTON_LEFT)
  {
    if(action == GLFW_PRESS)
    {
    }
  }
}

void OnScroll(GLFWwindow *window, double deltaX, double deltaY)
{
  gScrollY += deltaY;
}

void MouseInit(void)
{
  glfwSetCursorPos(gWindow, 0, 0);
  glfwSetScrollCallback(gWindow, OnScroll);
  glfwSetMouseButtonCallback(gWindow, OnClick);
}

void Update(float secondsElapsed)
{
  //keyboard
  const float moveSpeed = 0.5;
  if(glfwGetKey(gWindow, GLFW_KEY_K))
  {
    gCamera.offsetPosition(secondsElapsed * moveSpeed * glm::vec3(0, 1, 0));
  }else if(glfwGetKey(gWindow, GLFW_KEY_J))
  {
    gCamera.offsetPosition(secondsElapsed * moveSpeed * -glm::vec3(0, 1, 0));
  }

  if(glfwGetKey(gWindow, GLFW_KEY_H))
  {
    gCamera.offsetPosition(secondsElapsed * moveSpeed * -gCamera.right());
  }else if(glfwGetKey(gWindow, GLFW_KEY_L))
  {
    gCamera.offsetPosition(secondsElapsed * moveSpeed * gCamera.right());
  }

  if(glfwGetKey(gWindow, GLFW_KEY_Z))
  {
    gCamera.offsetPosition(secondsElapsed * moveSpeed * -gCamera.forward());
  }else if(glfwGetKey(gWindow, GLFW_KEY_X))
  {
    gCamera.offsetPosition(secondsElapsed * moveSpeed * gCamera.forward());
  }

  if(glfwGetKey(gWindow, GLFW_KEY_C))
  {
    CameraInit();
    gCamera.setFieldOfView(50.0);
    flag=true;
  }


  if(glfwGetKey(gWindow, GLFW_KEY_O))
  {
    wall_r-=1;
  }

  if(glfwGetKey(gWindow, GLFW_KEY_P))
  {
    wall_r+=1;
  }

  //mouse
  const float zoomSensitivity = -0.2f;
  float fieldOfView = gCamera.fieldOfView() + zoomSensitivity * (float)gScrollY;
  if(fieldOfView < 5.0f)
    fieldOfView = 5.0f;
  if(fieldOfView > 130.0f)
    fieldOfView = 130.0f;
  gCamera.setFieldOfView(fieldOfView);
  gScrollY = 0;

}

void CameraInit()
{
  gCamera.setPosition(glm::vec3(0, 0, 2.1));
  gCamera.setViewportAspectRatio(SIZE_X / SIZE_Y);
}


float CalcFPS(GLFWwindow *gWindow, float theTimeInterval, std::string theWindowTitle)
{
  static float t0Value = glfwGetTime();
  static int fpsFrameCount = 0;
  static float fps = 0.0;

  float currentTime = glfwGetTime();

  if(theTimeInterval < 0.1)
    theTimeInterval = 0.1;
  if(theTimeInterval > 10.0)
    theTimeInterval = 10.0;

  if((currentTime - t0Value) > theTimeInterval)
  {
    fps = (float)fpsFrameCount / (currentTime - t0Value);

    if(theWindowTitle != "NONE")
    {
      std::ostringstream stream;
      stream << fps;
      std::string fpsString = stream.str();

      theWindowTitle += " | FPS: " + fpsString;

      const char *pszConstString = theWindowTitle.c_str();
      glfwSetWindowTitle(gWindow, pszConstString);
    }else{
      std::cout << "FPS: " << fps << std::endl;
    }

    fpsFrameCount = 0;
    t0Value = glfwGetTime();
  }else{
    fpsFrameCount++;
  }
  return fps;
}

__device__ float d_clamp(float x, float a, float b)
{
  if (x < a){
    x = a;
  }
  if (x > b){
    x = b;
  }
  return x;
}


void PEC(GLubyte *h_g_data, float *ez, int X, int Y, int r){
  int index;
  /* for(int i=0;i<X;i++){ */
  /*   for(int j=0;j<Y/2-r/2;j++){ */
  /*     index = GRID_Y * j + i; */

  for(int i=0;i<X;i++){
    for(int j=0;j<Y/2-r/2;j++){
      index = GRID_Y * j + i;
      ez[index] = 0.0;
      h_g_data[index * 3 + 0]=(GLubyte)0;
      h_g_data[index * 3 + 1]=(GLubyte)0;
      h_g_data[index * 3 + 2]=(GLubyte)0;
    }
  }

  for(int i=0;i<X/2-r/2;i++){
    for(int j=Y/2+r/2;j<Y;j++){
      index = GRID_Y * j + i;
      ez[index] = 0.0;
      h_g_data[index * 3 + 0]=(GLubyte)0;
      h_g_data[index * 3 + 1]=(GLubyte)0;
      h_g_data[index * 3 + 2]=(GLubyte)0;
    }
  }

  for(int i=X/2+r/2;i<X;i++){
    for(int j=Y/2-r/2;j<Y;j++){
      index = GRID_Y * j + i;
      ez[index] = 0.0;
      h_g_data[index * 3 + 0]=(GLubyte)0;
      h_g_data[index * 3 + 1]=(GLubyte)0;
      h_g_data[index * 3 + 2]=(GLubyte)0;
    }
  }

  int j = Y/2-r/2;
  for(int i=X/2-r/2;i<X/2+r/2;i++){
    for(int k=Y/2-r/2;k<=j;k++){
      index = GRID_Y * k + i;
      ez[index] = 0.0;
      h_g_data[index * 3 + 0]=(GLubyte)0;
      h_g_data[index * 3 + 1]=(GLubyte)0;
      h_g_data[index * 3 + 2]=(GLubyte)0;
    }
    j+=(int)(sqrt(2));
  }

}

__device__ void GPU_PEC(GLubyte *h_g_data, float *ez, int X, int Y, int r){
  int i, j, index;
  i = blockDim.x * blockIdx.x + threadIdx.x;
  j = blockDim.y * blockIdx.y + threadIdx.y;
  int sqr = rsqrt(2.0);

  if(i<Y && j<X/2-r/2){
    index = Y * j + i;
    ez[index] = 0.0;
    h_g_data[index * 3 + 0]=(GLubyte)0;
    h_g_data[index * 3 + 1]=(GLubyte)0;
    h_g_data[index * 3 + 2]=(GLubyte)0;
  }
  if(i<Y/2-r/2 && j>= X/2+r/2 && j<X){
    index = Y * j + i;
    ez[index] = 0.0;
    h_g_data[index * 3 + 0]=(GLubyte)0;
    h_g_data[index * 3 + 1]=(GLubyte)0;
    h_g_data[index * 3 + 2]=(GLubyte)0;
  }
  if(i>=Y/2+r/2 && i<Y && j>=X/2-r/2 && j<X){
    index = Y * j + i;
    ez[index] = 0.0;
    h_g_data[index * 3 + 0]=(GLubyte)0;
    h_g_data[index * 3 + 1]=(GLubyte)0;
    h_g_data[index * 3 + 2]=(GLubyte)0;
  }

  if(i>=j){
    index = Y * j + i;
    ez[index] = 0.0;
    h_g_data[index * 3 + 0]=(GLubyte)0;
    h_g_data[index * 3 + 1]=(GLubyte)0;
    h_g_data[index * 3 + 2]=(GLubyte)0;
  }
}

__global__ void d_FDTD2d_tm_H(GLubyte *g_data, float *Ez, float *Hx, float *Hy, float *CHYX, float *CHYXL, float *CHXY, float *CHXYL, float *HXY, float *HYX, float *CHXLY, float *CHYLX, int L, unsigned int width, unsigned int height, float max, float min, float yellow, float green, float lightblue, int R)
{
  unsigned int i, j, index;
  float v;

  i = blockDim.x * blockIdx.x + threadIdx.x;
  j = blockDim.y * blockIdx.y + threadIdx.y;
  index = width * j + i;

  /*** range check ***/
  if ((i >= height) || (j >= width)){
    return;
  }

  /***create graphic data***/
  v = Ez[index];
  v = d_clamp(v, min, max);

  if(v > yellow) {
    g_data[index*3] = (GLubyte)255;
    g_data[index*3+1] = (GLubyte)(255-(v-yellow)/(max-yellow)*255);
    g_data[index*3+2] = (GLubyte)0;
  }else if(v > green){
    g_data[index*3] = (GLubyte)((v-green)/(yellow-green)*255);
    g_data[index*3+1] = (GLubyte)255;
    g_data[index*3+2] = (GLubyte)0;
  }else if(v > lightblue) {
    g_data[index*3] = (GLubyte)0;
    g_data[index*3+1] = (GLubyte)255;
    g_data[index*3+2] = (GLubyte)(255-(v-lightblue)/(green-lightblue)*255);
  }else{
    g_data[index*3] = (GLubyte)0;
    g_data[index*3+1] = (GLubyte)((v-min)/(lightblue-min)*255);
    g_data[index*3+2] = (GLubyte)255;
  }

  /*** range check (Hx)***/
  if ((i >= height-1) || (j >= width-1)){
    return;
  }
  //Hx
  Hx[index] = Hx[index] - (CHXLY[index]*(Ez[index+width]-Ez[index]));

  //Hx for PML
  if(i<L || i>width-L-1 || j<L || j>height-L-1){
    HXY[index]=CHXY[index]*HXY[index]-CHXYL[index]*(Ez[index+width]-Ez[index]);
    Hx[index]=HXY[index];
  }

  /*** range check (Hy)***/
  if (i == 0 || j == 0){
    return;
  }
  // Hy
  Hy[index] = Hy[index] + (CHYLX[index]*(Ez[index+1]-Ez[index]));

  //Hy for PML
  if(i<L || i>width-L-1 || j<L || j>height-L-1){
    HYX[index]=CHYX[index]*HYX[index]+CHYXL[index]*(Ez[index+1]-Ez[index]);
    Hy[index]=HYX[index];
  }
  
  GPU_PEC(g_data, Ez, width, height, R);
}

__global__ void d_FDTD2d_tm_E(float *Ez, float *Hx, float *Hy, float *CEZX, float *CEZXL, float *CEZY, float *CEZYL, float *EZX, float *EZY, float *CEZ, float *CEZLX, float *CEZLY, float step, unsigned int t, int L, unsigned int width, unsigned int height, int power_x, int power_y)
{
  unsigned int i, j, index;
  float pulse;

  i = blockDim.x * blockIdx.x + threadIdx.x;
  j = blockDim.y * blockIdx.y + threadIdx.y;
  index = width * j + i;

  /*** range check ***/
  if ((i >= height-1) || (j >= width-1) || i == 0 || j == 0){
    return;
  }

  pulse  =  sin((((t - 1)%(int)step)+1)*2.0*M_PI/step);

  //Ez
  if(i==power_x && j==power_y){
    Ez[index] = 1.0/376.7 * pulse;
  }else{
    Ez[index] = CEZ[index] * Ez[index] + CEZLX[index] * (Hy[index]-Hy[index-1]) - CEZLY[index] * (Hx[index]-Hx[index-width]);
  }

  if(i<L || (i>width-L-1) || j<L || (j>height-L-1)){
    EZX[index]=CEZX[index] * EZX[index] + CEZXL[index] * (Hy[index] - Hy[index-1]);
    EZY[index]=CEZY[index] * EZY[index] - CEZYL[index] * (Hx[index] - Hx[index-width]);
    Ez[index]=EZX[index]+EZY[index];
  }	
}

void launchGPUKernel(GLubyte *g_data, float *Ez, float *Hx, float *Hy, float *CEZX, float *CEZXL, float *CHYX, float *CHYXL, float *CEZY, float *CEZYL, float *CHXY, float *CHXYL, float *EZX, float *EZY, float *HXY, float *HYX, float *CEZ, float *CEZLX, float *CEZLY, float *CHXLY, float *CHYLX, float step, unsigned int t, int L, unsigned int width, unsigned int height, float max, float min, float yellow, float green, float lightblue, int power_x, int power_y, int R)
{
  dim3 grid(width / BLOCKDIM_X + 1, height / BLOCKDIM_Y + 1);
  dim3 block(BLOCKDIM_X, BLOCKDIM_Y, 1);


  d_FDTD2d_tm_E <<< grid, block >>> (Ez, Hx, Hy, CEZX, CEZXL, CEZY, CEZYL, EZX, EZY, CEZ, CEZLX, CEZLY, step, t, L, width, height, power_x, power_y);
  d_FDTD2d_tm_H <<< grid, block >>> (g_data, Ez, Hx, Hy, CHYX, CHYXL, CHXY, CHXYL, HXY, HYX, CHXLY, CHYLX, L, width, height, max, min, yellow, green, lightblue, R);
}


float h_clamp(float x, float a, float b)
{
  if (x < a)
    x = a;
  if (x > b)
    x = b;
  return x;
}

void h_FDTD2d_tm(GLubyte *g_data, float *Ez, float *Hx, float *Hy, float *CEZX, float *CEZXL, float *CHYX, float *CHYXL, float *CEZY, float *CEZYL, float *CHXY, float *CHXYL, float *EZX, float *EZY, float *HXY, float *HYX, float *CEZ, float *CEZLX, float *CEZLY, float *CHXLY, float *CHYLX, float step, unsigned int t, int L, unsigned int width, unsigned int height, float max, float min, float yellow, float green, float lightblue, int power_x, int power_y)
{
  unsigned int i, j, index;
  float pulse;
  pulse  =  sin((((t - 1)%(int)step)+1)*2.0*M_PI/step);

  //Ez
  for(j = 1; j < height-1; j++){
    for(i = 1; i < width-1; i++){
      index = width * j + i;
      if(i==power_x && j==power_y){
        Ez[index] = 1.0/376.7 * pulse;
      }else{
        Ez[index] = CEZ[index] * Ez[index] + CEZLX[index] * (Hy[index]-Hy[index-1]) - CEZLY[index] * (Hx[index]-Hx[index-width]);
      }
    }
  }

  /* Ez for PML */
  for(j = 1; j<height - 1; j++){
    for(i = 1; i<width - 1; i++){
      index = width * j + i;
      if(i<L || (i>width-L-1) || j<L || (j>height-L-1)){
        EZX[index]=CEZX[index] * EZX[index] + CEZXL[index] * (Hy[index] - Hy[index-1]);
        EZY[index]=CEZY[index] * EZY[index] - CEZYL[index] * (Hx[index] - Hx[index-width]);
        Ez[index]=EZX[index]+EZY[index];
      }
    }
  }

  // T=T+delta_t/2;

  //Hx
  for(j = 0; j<height - 1; j++){
    for(i = 1; i<width - 1; i++){
      index = width * j + i;
      Hx[index] = Hx[index] - (CHXLY[index]*(Ez[index+width]-Ez[index]));
    }
  }

  /* //Hx for PML*/
  for(j = 0; j<height - 1; j++){
    for(i = 1; i<width - 1; i++){
      index = width * j + i;
      if(i<L || i>width-L-1 || j<L || j>height-L-1){
        HXY[index]=CHXY[index]*HXY[index]-CHXYL[index]*(Ez[index+width]-Ez[index]);
        Hx[index]=HXY[index];
      }
    }
  }

  //Hy
  for(j = 1; j<height - 1; j++){
    for(i = 0; i<width - 1; i++){
      index = width * j + i;
      Hy[index] = Hy[index] + (CHYLX[index]*(Ez[index+1]-Ez[index]));
    }
  }

  //Hy for PML
  for(j = 1; j<height - 1; j++){
    for(i = 0; i<width - 1; i++){
      index = width * j + i;
      if(i<L || i>width-L-1 || j<L || j>height-L-1){
        HYX[index]=CHYX[index]*HYX[index]+CHYXL[index]*(Ez[index+1]-Ez[index]);
        Hy[index]=HYX[index];
      }
    }
  }

  /* Blank_Wall(g_data, Ez, wall_r, width, height); */

  // T=T+delta_t/2;


  /***create graphic data***/
  float v;
  for(j=0; j<height; j++){
    for(i=0; i<width; i++){
      index = width * j + i;
      v = Ez[index];
      v = h_clamp(v, min, max);

      if(v > yellow) {
        g_data[index*3] = (GLubyte)255;
        g_data[index*3+1] = (GLubyte)(255-(v-yellow)/(max-yellow)*255);
        g_data[index*3+2] = (GLubyte)0;
      }else if(v > Ez_green){
        g_data[index*3] = (GLubyte)((v-green)/(yellow-green)*255);
        g_data[index*3+1] = (GLubyte)255;
        g_data[index*3+2] = (GLubyte)0;
      }else if(v > lightblue) {
        g_data[index*3] = (GLubyte)0;
        g_data[index*3+1] = (GLubyte)255;
        g_data[index*3+2] = (GLubyte)(255-(v-lightblue)/(green-lightblue)*255);
      }else{
        g_data[index*3] = (GLubyte)0;
        g_data[index*3+1] = (GLubyte)((v-min)/(lightblue-min)*255);
        g_data[index*3+2] = (GLubyte)255;
      }
    }
  }

  PEC(g_data, Ez, width, height, wall_r);
}

void launchCPUKernel(GLubyte *g_data, float *Ez, float *Hx, float *Hy, float *CEZX, float *CEZXL, float *CHYX, float *CHYXL, float *CEZY, float *CEZYL, float *CHXY, float *CHXYL, float *EZX, float *EZY, float *HXY, float *HYX, float *CEZ, float *CEZLX, float *CEZLY, float *CHXLY, float *CHYLX, float step, unsigned int t, int L, unsigned int width, unsigned int height, float max, float min, float yellow, float green, float lightblue, int power_x, int power_y)
{
  h_FDTD2d_tm(g_data, Ez, Hx, Hy, CEZX, CEZXL, CHYX, CHYXL, CEZY, CEZYL, CHXY, CHXYL, EZX, EZY, HXY, HYX, CEZ, CEZLX, CEZLY, CHXLY, CHYLX, step, t, L, width, height, max, min, yellow, green, lightblue, power_x, power_y);

}


void setInitialData(unsigned int width, unsigned int height)
{
  lambda = c / freq;
  delta_x = lambda / resolution;
  delta_y = lambda / resolution;	
  delta_t = (1.0 / (sqrt(pow((1 / delta_x), 2.0)+pow((1 / delta_y), 2.0))))*(1.0 / c)*alpha;
  step = 1.0 / freq / delta_t;
  mu0 = 1.0e-7f * 4.0 * M_PI;
  ecmax = -(M+1)*epsilon0*c / (2.0*L*delta_x)*r0;
  Ez_range = Ez_max-Ez_min; // 2.7800852e-03f 
  Ez_yellow = Ez_range*0.75f+Ez_min;
  Ez_green = Ez_range*0.50f+Ez_min;
  Ez_lightblue = Ez_range*0.25f+Ez_min;
  wall_r = lambda / 2 / delta_x + 10;

  power_x = 12;
  power_y = GRID_Y/2 - 1;

  int i, j, index;
  float Z, ZZ;
  for(j = 0; j<GRID_Y; j++){
    for(i = 0; i<GRID_X; i++){
      index = GRID_X * j + i;
      h_mu_M[index]  =  mu0;
      h_epsilon_M[index] = epsilon0;
      h_sigma_M[index] = sigma0;
    }
  }

  for(j = 0; j<GRID_Y; j++){
    for(i = 0;i<GRID_X; i++){
      index = GRID_X * j + i;
      h_Ez[index] = 0.0;
      h_Hx[index] = 0.0;
      h_Hy[index] = 0.0;
      h_CEZX[index] = 0.0;
      h_CEZXL[index] = 0.0;
      h_CHYX[index] = 0.0;
      h_CHYXL[index] = 0.0;
      h_CEZY[index] = 0.0;
      h_CEZYL[index] = 0.0;
      h_CHXY[index] = 0.0;
      h_CHXYL[index] = 0.0;

      h_CEZ[index] = 0.0;
      h_CEZLX[index]=0.0;
      h_CEZLY[index]=0.0;
      h_CHXLY[index]=0.0;
      h_CHYLX[index]=0.0;
    }
  }

  for(i=0;i<GRID_X;i++){
    h_ECX[i]=0.0;
  }

  for(j=0;j<GRID_X;j++){
    h_ECY[j]=0.0;
  }

  for(i=0;i<L;i++){
    h_ECX[i] = ecmax * pow((L-i+0.5)/L,M);
    h_ECX[GRID_X-i-1] = h_ECX[i];
    h_ECY[i] = h_ECX[i];
    h_ECY[GRID_Y-i-1] = h_ECX[i];
  }

  //PML init
  for(i=0;i<GRID_X;i++){
    for(j=0;j<GRID_Y;j++){
      index = GRID_X * j + i;
      Z = (h_ECX[i] * delta_t)/(2.0*h_epsilon_M[index]);
      h_CEZX[index]=(1-Z)/(1+Z);
      h_CEZXL[index]=(delta_t/h_epsilon_M[index])/(1+Z)*(1.0/delta_x);
      h_CHYX[index]=(1-Z)/(1+Z);
      h_CHYXL[index]=(delta_t/h_mu_M[index])*(1.0/delta_x);
      Z = (h_ECY[j]*delta_t)/(2.0*h_epsilon_M[index]);
      h_CEZY[index]=(1-Z)/(1+Z);
      h_CEZYL[index]=(delta_t/h_epsilon_M[index])/(1+Z)*(1.0/delta_y);
      h_CHXY[index]=(1-Z)/(1+Z);
      h_CHXYL[index]=(delta_t/h_mu_M[index])*(1.0/delta_y);
    }
  }

  //FDTD init
  for(i=0;i<GRID_X;i++){
    for(j=0;j<GRID_Y;j++){
      index = GRID_X * j + i;
      ZZ = (h_sigma_M[index] * delta_t)/(2.0*h_epsilon_M[index]);
      h_CEZ[index]=(1-ZZ)/(1+ZZ);
      h_CEZLX[index]=(delta_t/h_epsilon_M[index])/(1+ZZ)*(1.0/delta_x);
      h_CEZLY[index]=(delta_t/h_epsilon_M[index])/(1+ZZ)*(1.0/delta_y);
      h_CHXLY[index]=delta_t/h_mu_M[index]*(1.0/delta_y);
      h_CHYLX[index]=delta_t/h_mu_M[index]*(1.0/delta_x);
    }
  }

  for(i=0;i<GRID_X;i++){
    for(j=0;j<GRID_Y;j++){
      index = GRID_X * j + i;
      h_g_data[index*3] = (GLubyte)0;
      h_g_data[index*3+1] = (GLubyte)0;
      h_g_data[index*3+2] = (GLubyte)0;	
    }
  }

  cudaMemcpy(d_Ez, h_Ez, sizeof(float) * GRID_Y * GRID_X, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Hx, h_Hx, sizeof(float) * GRID_Y * GRID_X, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Hy, h_Hy, sizeof(float) * GRID_Y * GRID_X, cudaMemcpyHostToDevice);
  cudaMemcpy(d_CEZX, h_CEZX, sizeof(float) * GRID_Y * GRID_X, cudaMemcpyHostToDevice);
  cudaMemcpy(d_CEZXL, h_CEZXL, sizeof(float) * GRID_Y * GRID_X, cudaMemcpyHostToDevice);
  cudaMemcpy(d_CHYX, h_CHYX, sizeof(float) * GRID_Y * GRID_X, cudaMemcpyHostToDevice);
  cudaMemcpy(d_CHYXL, h_CHYXL, sizeof(float) * GRID_Y * GRID_X, cudaMemcpyHostToDevice);
  cudaMemcpy(d_CEZY, h_CEZY, sizeof(float) * GRID_Y * GRID_X, cudaMemcpyHostToDevice);
  cudaMemcpy(d_CEZYL, h_CEZYL, sizeof(float) * GRID_Y * GRID_X, cudaMemcpyHostToDevice);
  cudaMemcpy(d_CHXY, h_CHXY, sizeof(float) * GRID_Y * GRID_X, cudaMemcpyHostToDevice);
  cudaMemcpy(d_CHXYL, h_CHXYL, sizeof(float) * GRID_Y * GRID_X, cudaMemcpyHostToDevice);
  cudaMemcpy(d_EZX, h_EZX, sizeof(float) * GRID_Y * GRID_X, cudaMemcpyHostToDevice);
  cudaMemcpy(d_EZY, h_EZY, sizeof(float) * GRID_Y * GRID_X, cudaMemcpyHostToDevice);
  cudaMemcpy(d_HXY, h_HXY, sizeof(float) * GRID_Y * GRID_X, cudaMemcpyHostToDevice);
  cudaMemcpy(d_HYX, h_HYX, sizeof(float) * GRID_Y * GRID_X, cudaMemcpyHostToDevice);
  cudaMemcpy(d_CEZ, h_CEZ, sizeof(float) * GRID_Y * GRID_X, cudaMemcpyHostToDevice);
  cudaMemcpy(d_CEZLX, h_CEZLX, sizeof(float) * GRID_Y * GRID_X, cudaMemcpyHostToDevice);
  cudaMemcpy(d_CEZLY, h_CEZLY, sizeof(float) * GRID_Y * GRID_X, cudaMemcpyHostToDevice);
  cudaMemcpy(d_CHXLY, h_CHXLY, sizeof(float) * GRID_Y * GRID_X, cudaMemcpyHostToDevice);
  cudaMemcpy(d_CHYLX, h_CHYLX, sizeof(float) * GRID_Y * GRID_X, cudaMemcpyHostToDevice);
}


void malloc_Initialdata(void)
{

  h_g_data = (GLubyte *)malloc(sizeof(GLubyte) * GRID_X * GRID_Y * 3);

  h_Ez  = (float *)malloc(sizeof(float) * GRID_Y * GRID_X);
  h_Hx  = (float *)malloc(sizeof(float) * GRID_Y * GRID_X);
  h_Hy  = (float *)malloc(sizeof(float) * GRID_Y * GRID_X);
  cudaMalloc((void**)&d_Ez, sizeof(float) *GRID_Y * GRID_X);
  cudaMalloc((void**)&d_Hx, sizeof(float) *GRID_Y * GRID_X);
  cudaMalloc((void**)&d_Hy, sizeof(float) *GRID_Y * GRID_X);

  h_sigma_M  = (float *)malloc(sizeof(float) * GRID_Y * GRID_X);
  h_epsilon_M = (float *)malloc(sizeof(float) * GRID_Y * GRID_X);
  h_mu_M = (float *)malloc(sizeof(float) * GRID_Y * GRID_X);

  h_ECX = (float *)malloc(sizeof(float) * GRID_X);
  h_ECY = (float *)malloc(sizeof(float) * GRID_Y);

  h_CEZX  = (float *)malloc(sizeof(float) * GRID_Y * GRID_X);
  h_CEZXL = (float *)malloc(sizeof(float) * GRID_Y * GRID_X);
  h_CHYX  = (float *)malloc(sizeof(float) * GRID_Y * GRID_X);
  h_CHYXL = (float *)malloc(sizeof(float) * GRID_Y * GRID_X);
  h_CEZY  = (float *)malloc(sizeof(float) * GRID_Y * GRID_X);
  h_CEZYL = (float *)malloc(sizeof(float) * GRID_Y * GRID_X);
  h_CHXY  = (float *)malloc(sizeof(float) * GRID_Y * GRID_X);
  h_CHXYL = (float *)malloc(sizeof(float) * GRID_Y * GRID_X);
  cudaMalloc((void**)&d_CEZX, sizeof(float) *GRID_Y * GRID_X);
  cudaMalloc((void**)&d_CEZXL, sizeof(float) *GRID_Y * GRID_X);
  cudaMalloc((void**)&d_CHYX, sizeof(float) *GRID_Y * GRID_X);
  cudaMalloc((void**)&d_CHYXL, sizeof(float) *GRID_Y * GRID_X);
  cudaMalloc((void**)&d_CEZY, sizeof(float) *GRID_Y * GRID_X);
  cudaMalloc((void**)&d_CEZYL, sizeof(float) *GRID_Y * GRID_X);
  cudaMalloc((void**)&d_CHXY, sizeof(float) *GRID_Y * GRID_X);
  cudaMalloc((void**)&d_CHXYL, sizeof(float) *GRID_Y * GRID_X);

  h_EZX = (float *)malloc(sizeof(float) * GRID_Y * GRID_X);
  h_EZY  = (float *)malloc(sizeof(float) * GRID_Y * GRID_X);
  h_HXY = (float *)malloc(sizeof(float) * GRID_Y * GRID_X);
  h_HYX = (float *)malloc(sizeof(float) * GRID_Y * GRID_X);
  cudaMalloc((void**)&d_EZX, sizeof(float) *GRID_Y * GRID_X);
  cudaMalloc((void**)&d_EZY, sizeof(float) *GRID_Y * GRID_X);
  cudaMalloc((void**)&d_HXY, sizeof(float) *GRID_Y * GRID_X);
  cudaMalloc((void**)&d_HYX, sizeof(float) *GRID_Y * GRID_X);

  h_CEZ  = (float *)malloc(sizeof(float) * GRID_Y * GRID_X);
  h_CEZLX = (float *)malloc(sizeof(float) * GRID_Y * GRID_X);
  h_CEZLY  = (float *)malloc(sizeof(float) * GRID_Y * GRID_X);
  h_CHXLY = (float *)malloc(sizeof(float) * GRID_Y * GRID_X);
  h_CHYLX  = (float *)malloc(sizeof(float) * GRID_Y * GRID_X);
  cudaMalloc((void**)&d_CEZ, sizeof(float) *GRID_Y * GRID_X);
  cudaMalloc((void**)&d_CEZLX, sizeof(float) *GRID_Y * GRID_X);
  cudaMalloc((void**)&d_CEZLY, sizeof(float) *GRID_Y * GRID_X);
  cudaMalloc((void**)&d_CHXLY, sizeof(float) *GRID_Y * GRID_X);
  cudaMalloc((void**)&d_CHYLX, sizeof(float) *GRID_Y * GRID_X);
}

void free_data(void)
{
  free(h_Ez);
  free(h_Hx);
  free(h_Hy);
  free(h_ECX);
  free(h_ECY);
  free(h_CEZX);
  free(h_CEZXL);
  free(h_CHYX);
  free(h_CHYXL);
  free(h_CEZY);
  free(h_CEZYL);
  free(h_CHXY);
  free(h_CHXYL);
  free(h_EZX);
  free(h_EZY);
  free(h_HXY);
  free(h_HYX);
  free(h_CEZ);
  free(h_CEZLX);
  free(h_CEZLY);
  free(h_CHXLY);
  free(h_CHYLX);
  free(h_sigma_M);
  free(h_epsilon_M);
  free(h_mu_M);

  cudaFree(d_Ez);
  cudaFree(d_Hx);
  cudaFree(d_Hy);
  cudaFree(d_CEZX);
  cudaFree(d_CEZXL);
  cudaFree(d_CHYX);
  cudaFree(d_CHYXL);
  cudaFree(d_CEZY);
  cudaFree(d_CEZYL);
  cudaFree(d_CHXY);
  cudaFree(d_CHXYL);
  cudaFree(d_EZX);
  cudaFree(d_EZY);
  cudaFree(d_HXY);
  cudaFree(d_HYX);
  cudaFree(d_CEZ);
  cudaFree(d_CEZLX);
  cudaFree(d_CEZLY);
  cudaFree(d_CHXLY);
  cudaFree(d_CHYLX);
}














void RunGPUKernel(void){
  cudaGraphicsMapResources(1, &pbo_res, 0);
  cudaGraphicsResourceGetMappedPointer((void**)&d_g_data, NULL, pbo_res);

  if(!flag){
    launchGPUKernel(d_g_data, d_Ez, d_Hx, d_Hy, d_CEZX, d_CEZXL, d_CHYX, d_CHYXL, d_CEZY, d_CEZYL, d_CHXY, d_CHXYL, d_EZX, d_EZY, d_HXY, d_HYX, d_CEZ, d_CEZLX, d_CEZLY, d_CHXLY, d_CHYLX, step, kt, L, GRID_X, GRID_Y, Ez_max, Ez_min, Ez_yellow, Ez_green, Ez_lightblue, power_x, power_y, wall_r);
  }
  kt++;

  cudaGraphicsUnmapResources(1, &pbo_res, 0);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gPBO);
  glBindTexture(GL_TEXTURE_2D, gTEX);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, GRID_X, GRID_Y, GL_RGB, GL_UNSIGNED_BYTE, NULL);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void RunCPUKernel(void){

  if(!flag){
    launchCPUKernel(h_g_data, h_Ez, h_Hx, h_Hy, h_CEZX, h_CEZXL, h_CHYX, h_CHYXL, h_CEZY, h_CEZYL, h_CHXY, h_CHXYL, h_EZX, h_EZY, h_HXY, h_HYX, h_CEZ, h_CEZLX, h_CEZLY, h_CHXLY, h_CHYLX, step, kt, L, GRID_X, GRID_Y, Ez_max, Ez_min, Ez_yellow, Ez_green, Ez_lightblue, power_x, power_y);
  }


  kt++;

  glBindTexture(GL_TEXTURE_2D, gTEX);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, GRID_X, GRID_Y, GL_RGB, GL_UNSIGNED_BYTE, h_g_data);
}
void InitPBO(GLuint *pbo, unsigned int size, struct cudaGraphicsResource **pbo_res, unsigned int pbo_res_flags){
  glGenBuffers(1, pbo);
  glBindBuffer(GL_ARRAY_BUFFER, *pbo);
  glBufferData(GL_ARRAY_BUFFER, size, NULL, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  cudaGraphicsGLRegisterBuffer(pbo_res, *pbo, pbo_res_flags);
}

void InitTexData(){
  int i, j;
  for(i=0;i<GRID_X;i++){
    for(j=0;j<GRID_Y;j++){
      int index = i*GRID_X+j;
      h_g_data[index*3+0] = (GLubyte)i;
      h_g_data[index*3+1] = (GLubyte)GRID_X-i;
      h_g_data[index*3+2] = (GLubyte)GRID_X-i;
    }
  }
  glGenTextures(1, &gTEX);
  glBindTexture(GL_TEXTURE_2D, gTEX);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, GRID_X, GRID_Y, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
}

void LoadShaders(){
  std::vector<tdogl::Shader::Shader> shaders;
  shaders.push_back(tdogl::Shader::shaderFromFile("./vertex-shader.glsl", GL_VERTEX_SHADER));
  shaders.push_back(tdogl::Shader::shaderFromFile("./fragment-shader.glsl", GL_FRAGMENT_SHADER));
  gProgram = new tdogl::Program(shaders);
}

// loads a triangle into the VAO global
void LoadTriangle() {
  // make and bind the VAO
  glGenVertexArrays(1, &gVAO);
  glBindVertexArray(gVAO);

  // make and bind the VBO
  glGenBuffers(1, &gVBO);
  glBindBuffer(GL_ARRAY_BUFFER, gVBO);

  GLfloat vertexData[] = {
    1.0, -1.0, 0.0f, 1.0f, 0.0f,
    -1.0, -1.0, 0.0f, 0.0f, 0.0f,
    1.0,  1.0, 0.0f, 1.0f, 1.0f,

    1.0,  1.0, 0.0f, 1.0f, 1.0f,
    -1.0, -1.0, 0.0f, 0.0f, 0.0f,
    -1.0,  1.0, 0.0f, 0.0f, 1.0f

  };
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertexData), vertexData, GL_STATIC_DRAW);

  // connect the xyz to the "vert" attribute of the vertex shader
  glEnableVertexAttribArray(gProgram->attrib("vert"));
  glVertexAttribPointer(gProgram->attrib("vert"), 3, GL_FLOAT, GL_FALSE, 5*sizeof(GLfloat), NULL);

  glEnableVertexAttribArray(gProgram->attrib("verTexCoord"));
  glVertexAttribPointer(gProgram->attrib("verTexCoord"), 2, GL_FLOAT, GL_FALSE, 5*sizeof(GLfloat), (const GLvoid*)(3*sizeof(GLfloat)));

  // unbind the VBO and VAO
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
}


// draws a single frame
void Render() {
  // clear everything
  glClearColor(255, 255, 255, 1); // black
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // bind the program (the shaders)
  glUseProgram(gProgram->object());

  gProgram->setUniform("camera", gCamera.matrix());

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, gTEX);
  glUniform1f(gProgram->uniform("tex"), 0);

  // bind the VAO (the triangle)
  glBindVertexArray(gVAO);

  // draw the VAO
  glDrawArrays(GL_TRIANGLES, 0, 6);


  // unbind the VAO
  glBindVertexArray(0);

  glBindTexture(GL_TEXTURE_2D, 0);
  // unbind the program
  glUseProgram(0);

}

void OnError(int errorCode, const char* msg) {
  throw std::runtime_error(msg);
}

// the program starts here
void AppMain() {
  // initialise GLFW
  glfwSetErrorCallback(OnError);
  if(!glfwInit())
    throw std::runtime_error("glfwInit failed");

  // open a window with GLFW
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
  gWindow = glfwCreateWindow(SIZE_X, SIZE_Y, "Strting...", NULL, NULL);
  if(!gWindow)
    throw std::runtime_error("glfwCreateWindow failed. Can your hardware handle OpenGL 3.2?");

  // GLFW settings
  glfwMakeContextCurrent(gWindow);

  // initialise GLEW
  glewExperimental = GL_TRUE; //stops glew crashing on OSX :-/
  if(glewInit() != GLEW_OK)
    throw std::runtime_error("glewInit failed");

  // print out some info about the graphics drivers
  std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
  std::cout << "GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
  std::cout << "Vendor: " << glGetString(GL_VENDOR) << std::endl;
  std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;

  // make sure OpenGL version 3.2 API is available
  if(!GLEW_VERSION_3_2)
    throw std::runtime_error("OpenGL 3.2 API is not available.");







  malloc_Initialdata();

  setInitialData(GRID_X, GRID_Y);

  LoadShaders();



  InitPBO(&gPBO, sizeof(GLubyte)*GRID_X*GRID_Y*3, &pbo_res, cudaGraphicsRegisterFlagsWriteDiscard);

  // load vertex and fragment shaders into opengl
  InitTexData();  

  // create buffer and fill it with the points of the triangle
  LoadTriangle();

  CameraInit();

  MouseInit();

  ctx = nk_glfw3_init(gWindow, NK_GLFW3_INSTALL_CALLBACKS);
  {
    struct nk_font_atlas *atlas;
    nk_glfw3_font_stash_begin(&atlas);
    nk_glfw3_font_stash_end();
  }


  float lastTime = glfwGetTime();
  // run while the window is open
  while(!glfwWindowShouldClose(gWindow)){
    // process pending events
    glfwPollEvents();

    nk_glfw3_new_frame();

    /* RunCPUKernel(); */
    RunGPUKernel();
    // draw one frame
    Render();

    CalcFPS(gWindow, 1.0, "GL3.2 FDTD2D_TM");

    float thisTime = glfwGetTime();
    Update((float)(thisTime - lastTime));

    
    GUIRender(ctx, GRID_X, GRID_Y);
    nk_glfw3_render(NK_ANTI_ALIASING_ON, MAX_VERTEX_BUFFER, MAX_ELEMENT_BUFFER);


    // swap the display buffers (displays what was just drawn)
    glfwSwapBuffers(gWindow);

    
    if(glfwGetKey(gWindow, GLFW_KEY_ESCAPE)){
      free_data();
      glDeleteTextures(1, &gTEX);
      cudaGraphicsUnregisterResource(pbo_res);
      glDeleteBuffers(1, &gPBO);
      cudaFree(d_g_data);
      free(h_g_data);
      cudaDeviceReset();
      glfwSetWindowShouldClose(gWindow, GL_TRUE);
    }

  }
  // clean up and exit
  nk_glfw3_shutdown();
  glfwTerminate();
}

int main(int argc, char *argv[]) {
  try {
    AppMain();
  } catch (const std::exception& e){
    std::cerr << "ERROR: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

