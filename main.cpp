#include <cassert>
#include <iostream>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <random>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "./glm/glm.hpp"

#include "Program.h"
#include "Camera.h"

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

const glm::vec2 SCREEN_SIZE(1080, 1080);
const glm::vec2 WINDOW_POS(128, 128);
const glm::vec2 GRID_SIZE(256, 256);

//texture
GLuint tex;
GLubyte *h_g_data;

//window
GLFWwindow *gWindow = NULL;
//GLSL
tdogl::Program *gProgram = NULL;
// tdogl::Texture *gTexture = NULL;
tdogl::Camera gCamera;

GLuint gVAO = 0;
GLuint gVBO = 0;
float gScrollY = 0.0;

/****************
 *   parameter   *
 ****************/
const float c = 2.99792458e8;
const float freq = 1.0e9;/* float freq = 1.0e15; */
float lambda;
const float resolution = 20.0;/* float resolution = 40.0; */
float delta_x;
float delta_y;
const float alpha = 0.5;
float delta_t;
float step;
float mu0;
const float sigma0 = 0;
const float epsilon0 = 8.854187e-12f;
const int M = 4;
const int L = 12;
/* int L = 24; */
float r0 = -6;
float ecmax;
float Ez_max = 2.060459378159e-03f;
float Ez_min = -2.060459378159e-03f;
// float Ez_min = -7.196258220476e-04f;
float Ez_range;
float Ez_yellow;
float Ez_green;
float Ez_lightblue;

// float pulse;
float T = 0.0;

float **Ez, **Hx, **Hy, **sigma_M, **epsilon_M, **mu_M;
float *ECX, *ECY;
float **CEZX, **CEZXL, **CHYX, **CHYXL, **CEZY, **CEZYL, **CHXY, **CHXYL;
float **EZX, **EZY, **HXY, **HYX;

float **CEZ, **CEZLX, **CEZLY, **CHXLY, **CHYLX;


// counter
int kt = 1;
int incrt = 1;

float anim_time = 0.0f;
float anim_dt;

bool flag = true;
bool z_flag = false;

// frame rate
int GLframe = 0;
int GLtimenow = 0;
int GLtimebase = 0;

//rect
float rectD;

int power_x, power_y;


//sampling mode
// bool sampling_flag=false;
// bool sampling_pick=false;
// double sampling[2]={0.0,0.0};
// float *sampling_list;
// int sampling_len=0;

float E_Max, E_Min;
/**************************
 *   forward declaration   *
 **************************/

void OnError(int errorCode, const char *msg);
void AppMain(void);
void LoadShaders(void);
void Render(void);
void AllocInit(void);
void AllocFree(void);
void ParamInit(void);
// void PrintInfo(void);
void FDTDInit(void);
void PMLInit(void);
void TextureInit(void);
void CreateTexture(GLuint *tex);
void DeleteTexture(GLuint *tex);
void RunCPUKernel(void);
void LaunchCPUKernel(void);
float Compare(float x, float a, float b);
void GLFW_GLEWInit(void);
float CalcFPS(GLFWwindow *gWindow, float theTimeInterval = 1.0, std::string theWindowTitle = "NONE");
void CameraInit(void);
void Update(float secondsElapsed);
// void CheckOpenGLError(void);
void OnScroll(GLFWwindow *window, double deltaX, double deltaY);
void MouseInit(void);
void GUIInit(struct nk_context *ctx, GLFWwindow *gWindow);
void GUIRender(struct nk_context *ctx);
void OnClick(GLFWwindow *window, int button, int action, int mods);
double mapping(double value, double istart, double istop, double ostart, double ostop);
// unsigned ftoi(double d);
void CPU_EZ(float **ez, float **cez, float **cezlx, float **hy, float **cezly, float **hx, int gx, int gy);
void CPU_HX(float **hx, float **chxly, float **ez, int gc, int gy);
void CPU_HY(float **hy, float **chylx, float **ez, int gx, int gy);
void CPU_PML_EZ(float **ezx, float **cezx, float **cezxl, float **hy, float **ezy, float **cezy, float **cezyl, float **hx, float **ez, int gx, int gy, int l);
void CPU_PML_HX(float **hxy, float **chxy, float **chxyl, float **ez, float **hx, int gx, int gy, int l);
void CPU_Input(float **ez, int px, int py, float p, int gx, int gy);
void CPU_PML_HY(float **hyx, float **chyx, float **chyxl, float **ez, float **hy, int gx, int gy, int l);
void CPU_MAX_MIN(float **ez, float *max, float *min, int gx, int gy);
void CPU_Create_Data(GLubyte *data, float **ez, int gx, int gy, float yellow, float green, float blue, float max, float min);
void Blank_Wall(float **ez, int gx, int gy, int R, GLubyte *data);
void FDTD2dTM(float **_Ez, float **_Hx, float **_Hy,
              float **_CEZ, float **_CEZLX, float **_CEZLY, float **_CHXLY, float **_CHYLX,
              float **_CEZX, float **_CEZXL, float **_CHYX, float **_CHYXL, float **_CEZY, float **_CEZYL, float **_CHXY, float **_CHXYL,
              float **_EZX, float **_EZY, float **_HXY, float **_HYX, 
              float _step, int *_kt, glm::vec2 _GRID_SIZE, float *_T,
              int _power_x, int _power_y, float *_Ez_max, float *_Ez_min,
              GLubyte *_h_g_data, float _rectD, float _L,
              float _Ez_yellow, float _Ez_green, float _Ez_lightblue,
              float _delta_t);

void Blank_Wall(float **ez, int gx, int gy, int R, GLubyte *data){
  //blank
  int index;
  for(int i=0; i<gy ; i++){
    for(int j=0 ; j<gx/2-R/2 ; j++){
      ez[i][j] = 0.0;
      index = gy * j + i;
      data[index*3+0] = (GLubyte)0;
      data[index*3+1] = (GLubyte)0;
      data[index*3+2] = (GLubyte)0;
    }
  }
  for(int i=0 ; i<gy/2-R/2 ; i++){
    for(int j=gx/2+R/2 ; j<gx ; j++){
      ez[i][j] = 0.0;
      index = gy * j + i;
      data[index*3+0] = (GLubyte)0;
      data[index*3+1] = (GLubyte)0;
      data[index*3+2] = (GLubyte)0;
    }
  }
  for(int i=gy/2+R/2 ; i<gy ; i++){
    for(int j=gx/2-R/2 ; j<gx ; j++){
      ez[i][j] = 0.0;
      index = gy * j + i;
      data[index*3+0] = (GLubyte)0;
      data[index*3+1] = (GLubyte)0;
      data[index*3+2] = (GLubyte)0;
    }
  }
  int j=gx/2-R/2;
  for(int i=gy/2-R/2 ; i<gy/2+R/2 ; i++, j+=(int)sqrt(2)){
    for(int k=gx/2-R/2 ; k<=j ; k++){
      ez[j][k] = 0.0;
      index = gy * k + i;
      data[index*3+0] = (GLubyte)0;
      data[index*3+1] = (GLubyte)0;
      data[index*3+2] = (GLubyte)0;
    }
  }
  //wall
  for(int i=0;i<gy;i++)
  {
    int j=0;
    ez[i][j]=0.0;
    index = gy * j + i;
    data[index*3+0] = (GLubyte)0;
    data[index*3+1] = (GLubyte)0;
    data[index*3+2] = (GLubyte)0;
    j=gx-1;
    ez[i][j]=0.0;
    index = gy * j + i;
    data[index*3+0] = (GLubyte)0;
    data[index*3+1] = (GLubyte)0;
    data[index*3+2] = (GLubyte)0;
  }
  for(int j=0;j<gx;j++)
  {
    int i=0;
    ez[i][j]=0.0;
    index = gy * j + i;
    data[index*3+0] = (GLubyte)0;
    data[index*3+1] = (GLubyte)0;
    data[index*3+2] = (GLubyte)0;
    i=gy-1;
    ez[i][j]=0.0;
    index = gy * j + i;
    data[index*3+0] = (GLubyte)0;
    data[index*3+1] = (GLubyte)0;
    data[index*3+2] = (GLubyte)0;
  }
}

void CPU_Create_Data(GLubyte *data, float **ez, int gx, int gy, float yellow, float green, float blue, float *max, float *min){
  for(int j=0 ; j<gx ; j++){
    for(int i=0 ; i<gy ; i++){
      int index = gy * j + i;
      float v = ez[i][j];
      if(v > yellow){
        float d = mapping(v, yellow, *max, 0, 255);
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
        float d = mapping(v, *min, blue, 255, 0);
        data[index * 3 + 0] = (GLubyte)0;
        data[index * 3 + 1] = (GLubyte)d;
        data[index * 3 + 2] = (GLubyte)255;
      }
    }
  }
}

void CPU_MAX_MIN(float **ez, float *max, float *min, int gx, int gy){
  for(int j=1 ; j<gx-1 ; j++){
    for(int i=0 ; i<gy-1 ; i++){
      if(ez[i][j] > *max){
        *max = ez[i][j];
      }
      if(ez[i][j] < *min){
        *min = ez[i][j];
      }
    }
  }
}

void CPU_PML_HY(float **hyx, float **chyx, float **chyxl, float **ez, float **hy, int gx, int gy, int l){
  for(int j=1 ; j<gx-1 ; j++){
    for(int i=0 ; i<gy-1 ; i++){
      if(i<l || i>gy-l-1 || j<l || j>gx-l-1){
        hyx[i][j] = chyx[i][j] * hyx[i][j] + chyxl[i][j] * (ez[i+1][j]-ez[i][j]);
        hy[i][j] = hyx[i][j];
      }
    }
  }
}

void CPU_Input(float **ez, int px, int py, float p, int gx, int gy){
  for(int j=1 ; j<gx-1 ; j++){
    for(int i=1 ; i<gy-1 ; i++){
      if( i==px && j==py ){
        ez[i][j] = 1.0/376.7 * p;
      }
    }
  }
}

void CPU_PML_HX(float **hxy, float **chxy, float **chxyl, float **ez, float **hx, int gx, int gy, int l){
  for(int j=0 ; j<gx-1 ; j++){
    for(int i=1 ; i<gy-1 ; i++){
      if(i<l || i>gy-l-1 || j<l || j>gx-l-1){
        hxy[i][j] = chxy[i][j] * hxy[i][j] - chxyl[i][j] * (ez[i][j+1]-ez[i][j]);
        hx[i][j] = hxy[i][j];
      }
    }
  }
}

void CPU_PML_EZ(float **ezx, float **cezx, float **cezxl, float **hy, float **ezy, float **cezy, float **cezyl, float **hx, float **ez, int gx, int gy, int l){
  for(int j=1 ; j<gx-1 ; j++){
    for(int i=1 ; i<gy-1 ; i++){
      if(i<l || (i>gy-l-1) || j<l || (j>gx-l-1)){
        ezx[i][j] = cezx[i][j] * ezx[i][j] + cezxl[i][j] * (hy[i][j] - hy[i-1][j]);
        ezy[i][j] = cezy[i][j] * ezy[i][j] - cezyl[i][j] * (hx[i][j] - hx[i][j-1]);
        ez[i][j] = ezx[i][j] + ezy[i][j];
      }
    }
  }
}

void CPU_EZ(float **ez, float **cez, float **cezlx, float **hy, float **cezly, float **hx, int gx, int gy){
  for(int j=1 ; j<gx-1 ; j++){
    for(int i=1 ; i<gy-1 ; i++){
      ez[i][j] = cez[i][j] * ez[i][j] + cezlx[i][j] *(hy[i][j]-hy[i-1][j]) - cezly[i][j] * (hx[i][j] - hx[i][j-1]);
    }
  }
}

void CPU_HX(float **hx, float **chxly, float **ez, int gx, int gy){
  for(int j=0 ; j<gx-1 ; j++){
    for(int i=1 ; i<gy-1 ; i++){
      hx[i][j] = hx[i][j] - (chxly[i][j]*(ez[i][j+1]-ez[i][j]));
    }
  }
}

void CPU_HY(float **hy, float **chylx, float **ez, int gx, int gy){
  for(int j = 1; j<gx-1 ; j++){
    for(int i=0 ; i<gy-1 ; i++){
      hy[i][j]=hy[i][j]+(chylx[i][j]*(ez[i+1][j]-ez[i][j]));
    }
  }
}

// unsigned ftoi(double d)
// {
//   d += 4503599627370496.0;
//   return (unsigned &)d;
// }

double mapping(double value, double istart, double istop, double ostart, double ostop)
{
  return ostart + (ostop - ostart) * ((value - istart) / (istop - istart));
}

void OnClick(GLFWwindow *window, int button, int action, int mods)
{
  if(button == GLFW_MOUSE_BUTTON_LEFT)
  {
    if(action == GLFW_PRESS)
    {
      // if(sampling_pick==true)
      // {
      //   double mouseX, mouseY;
      //   double X, Y;
      //   glfwGetCursorPos(window, &mouseX, &mouseY);
      //   X=mapping(mouseX, 0, SCREEN_SIZE.x, 0, GRID_SIZE.x);
      //   Y=mapping(mouseY, 0, SCREEN_SIZE.y, 0, GRID_SIZE.y);
      //   {
      //     sampling[0]=X;
      //     sampling[1]=Y;
      //   }
      //   sampling_pick=false;
      // }
    }
  }
}

void GUIRender(struct nk_context *ctx)
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
      FDTDInit();
      PMLInit();
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

    // if (nk_tree_push(ctx, NK_TREE_NODE, "Sampling", NK_MINIMIZED))
    // {
    //   int i;
    //   static int onoff=nk_false;
    //   static int pick=nk_false;
    //   nk_layout_row_dynamic(ctx, 20, 4);
    //   nk_checkbox_label(ctx, "On/Off", &onoff);
    //   nk_selectable_label(ctx, "Pick", NK_TEXT_LEFT, &pick);
    //   nk_value_int(ctx, "x", sampling[0]);
    //   nk_value_int(ctx, "y", ftoi(GRID_SIZE.y)-sampling[1]);
    //   if(onoff==nk_false)
    //     sampling_flag=false;
    //   else
    //     sampling_flag=true;
    //
    //   if(pick==nk_false)
    //     sampling_pick=false;
    //   else
    //     sampling_pick=true;
    //
    //   nk_layout_row_dynamic(ctx, 60, 1);
    //
    //   if(nk_chart_begin(ctx, NK_CHART_LINES, 100, Ez_min, Ez_max))
    //   {
    //     for(i=0;i<100;i++)
    //     {
    //       nk_chart_push(ctx, sampling_list[i]);
    //     }
    //     nk_chart_end(ctx);
    //   }
    //
    //   nk_tree_pop(ctx);
    // }

    if(nk_tree_push(ctx, NK_TREE_NODE, "Shape", NK_MINIMIZED))
    {
      nk_layout_row_begin(ctx, NK_STATIC, 20, 3);
      nk_layout_row_push(ctx, 20);
      nk_button_set_behavior(ctx, NK_BUTTON_REPEATER);
      if(nk_button_symbol(ctx, NK_SYMBOL_TRIANGLE_LEFT)){
        rectD-=0.1;
        if(rectD<=0.0){
          rectD=0.0;
        }
      }

      nk_layout_row_push(ctx, 100);
      nk_value_float(ctx, "Width", rectD);

      nk_layout_row_push(ctx, 20);
      nk_button_set_behavior(ctx, NK_BUTTON_REPEATER);
      if(nk_button_symbol(ctx, NK_SYMBOL_TRIANGLE_RIGHT)){
        rectD+=0.1;
      }
      nk_layout_row_end(ctx);
      nk_tree_pop(ctx);
    }

  }
  nk_end(ctx);

}

void MouseInit(void)
{
  glfwSetCursorPos(gWindow, 0, 0);
  glfwSetScrollCallback(gWindow, OnScroll);
  glfwSetMouseButtonCallback(gWindow, OnClick);
}

void OnScroll(GLFWwindow *window, double deltaX, double deltaY)
{
  gScrollY += deltaY;
}

// void CheckOpenGLError(void)
// {
//   GLenum error = glGetError();
//   if(error != GL_NO_ERROR)
//     std::cerr << "OpenGL Error: " << error << std::endl;
// }
//
void Update(float secondsElapsed)
{
  //keyboard
  const float moveSpeed = 2.0;
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
  gCamera.setViewportAspectRatio(SCREEN_SIZE.x / SCREEN_SIZE.y);
}

// float CalcFPS(GLFWwindow *gWindow, float theTimeInterval, std::string theWindowTitle)
// {
//   static float t0Value = glfwGetTime();
//   static int fpsFrameCount = 0;
//   static float fps = 0.0;
//
//   float currentTime = glfwGetTime();
//
//   if(theTimeInterval < 0.1)
//     theTimeInterval = 0.1;
//   if(theTimeInterval > 10.0)
//     theTimeInterval = 10.0;
//
//   if((currentTime - t0Value) > theTimeInterval)
//   {
//     fps = (float)fpsFrameCount / (currentTime - t0Value);
//
//     if(theWindowTitle != "NONE")
//     {
//       std::ostringstream stream;
//       stream << fps;
//       std::string fpsString = stream.str();
//
//       theWindowTitle += " | FPS: " + fpsString;
//
//       const char *pszConstString = theWindowTitle.c_str();
//       glfwSetWindowTitle(gWindow, pszConstString);
//     }else{
//       std::cout << "FPS: " << fps << std::endl;
//     }
//
//     fpsFrameCount = 0;
//     t0Value = glfwGetTime();
//   }else{
//     fpsFrameCount++;
//   }
//   return fps;
// }

void GLFW_GLEWInit()
{
  glfwSetErrorCallback(OnError);
  if(!glfwInit())
    throw std::runtime_error("glfwInit faild");

  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

  gWindow = glfwCreateWindow((int)SCREEN_SIZE.x, (int)SCREEN_SIZE.y, "GL3.2 FDTD2D_TM", NULL, NULL);
  if(!gWindow)
    throw std::runtime_error("glfwCreateWindow faild");

  glfwMakeContextCurrent(gWindow);
  glewExperimental = GL_TRUE;
  if(glewInit() != GLEW_OK)
    throw std::runtime_error("glewInit failed");
  if(!GLEW_VERSION_3_2)
    throw std::runtime_error("OpenGL 3.2 API not found");
}

float Compare(float x, float a, float b)
{
  if(x < a)
    x = a;
  if(x > b)
    x = b;
  return x;
}

void FDTD2dTM(float **_Ez, float **_Hx, float **_Hy,
              float **_CEZ, float **_CEZLX, float **_CEZLY, float **_CHXLY, float **_CHYLX,
              float **_CEZX, float **_CEZXL, float **_CHYX, float **_CHYXL, float **_CEZY, float **_CEZYL, float **_CHXY, float **_CHXYL,
              float **_EZX, float **_EZY, float **_HXY, float **_HYX, 
              float _step, int *_kt, glm::vec2 _GRID_SIZE, float *_T,
              int _power_x, int _power_y, float *_Ez_max, float *_Ez_min,
              GLubyte *_h_g_data, float _rectD, float _L,
              float _Ez_yellow, float _Ez_green, float _Ez_lightblue,
              float _delta_t)
{
  float pulse;
  pulse  =  sin(((( *_kt - 1)%(int)_step)+1)*2.0*M_PI/_step);

  //Ez
  CPU_EZ( _Ez, _CEZ, _CEZLX, _Hy, _CEZLY, _Hx, (int)_GRID_SIZE.x, (int)_GRID_SIZE.y);

  /* Ez for PML */
  CPU_PML_EZ( _EZX, _CEZX, _CEZXL, _Hy, _EZY, _CEZY, _CEZYL, _Hx, _Ez, (int)_GRID_SIZE.x, (int)_GRID_SIZE.y, L);

  //power input
  CPU_Input( _Ez, _power_x, _power_y, pulse, (int)_GRID_SIZE.x, (int)_GRID_SIZE.y);

  //Hx
  CPU_HX( _Hx, _CHXLY, _Ez, (int)_GRID_SIZE.x, (int)_GRID_SIZE.y);

  /* //Hx for PML*/
  CPU_PML_HX( _HXY, _CHXY, _CHXYL, _Ez, _Hx, (int)_GRID_SIZE.x, (int)_GRID_SIZE.y, L);

  //Hy
  CPU_HY( _Hy, _CHYLX, _Ez, (int)_GRID_SIZE.x, (int)_GRID_SIZE.y);

  //Hy for PML
  CPU_PML_HY( _HYX, _CHYX, _CHYXL, _Ez, _Hy, (int)_GRID_SIZE.x, (int)_GRID_SIZE.y, _L);

  //find max min
  CPU_MAX_MIN( _Ez, _Ez_max, _Ez_min, (int)_GRID_SIZE.x, (int)_GRID_SIZE.y);

  /***create graphic data***/
  CPU_Create_Data( _h_g_data, _Ez, (int)_GRID_SIZE.x, (int)_GRID_SIZE.y, _Ez_yellow, _Ez_green, _Ez_lightblue, _Ez_max, _Ez_min);

  Blank_Wall( _Ez, (int)_GRID_SIZE.x, (int)_GRID_SIZE.y, _rectD, _h_g_data);

  // if(sampling_pick==true)
  // {
  //   int x=ftoi(sampling[0]);
  //   int y=ftoi(sampling[1]);
  //   index=(int)GRID_SIZE.y * ((int)GRID_SIZE.y-y)+x;
  //   h_g_data[index*3] = (GLubyte)255;
  //   h_g_data[index*3+1] = (GLubyte)0;
  //   h_g_data[index*3+2] = (GLubyte)0;
  //
  //   if(sampling_len>=100)
  //     sampling_len=0;
  //   sampling_list[sampling_len]=Ez[x][ftoi(GRID_SIZE.y)-y];
  //   sampling_len++;
  // }

  *_kt = *_kt+1;
  *_T = *_T+delta_t;
}

void LaunchCPUKernel()
{
  FDTD2dTM(Ez, Hx, Hy,
           CEZ, CEZLX, CEZLY, CHXLY, CHYLX,
           CEZX, CEZXL, CHYX, CHYXL, CEZY, CEZYL, CHXY, CHXYL,
           EZX, EZY, HXY, HYX,
           step, &kt, GRID_SIZE, &T,
           power_x, power_y, &Ez_max, &Ez_min,
           h_g_data, rectD, L,
           Ez_yellow, Ez_green, Ez_lightblue,
           delta_t);
}

void RunCPUKernel()
{
  if(flag==true)
    LaunchCPUKernel();

  glBindTexture(GL_TEXTURE_2D, tex);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, GRID_SIZE.x, GRID_SIZE.y, GL_RGB, GL_UNSIGNED_BYTE, h_g_data);
}

void DeleteTexture(int index, GLuint *tex)
{
  glDeleteTextures(index, tex);
  *tex=0;
}

void CreateTexture(int index, GLuint *tex)
{
  glGenTextures(index, tex);
  glBindTexture(GL_TEXTURE_2D, *tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, GRID_SIZE.x, GRID_SIZE.y, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
}

void TextureInit()
{
  glClearColor(135.0f/255.0f, 206.0f/255.0f, 250.0f/255.0f, 1.0f);
  glEnable(GL_TEXTURE_2D);

  h_g_data = (GLubyte *)malloc(sizeof(GLubyte) * GRID_SIZE.x * GRID_SIZE.y * 3);

  for(int i=0; i<GRID_SIZE.x; i++)
    for(int j=0; j<GRID_SIZE.y; j++)
    {
      int index = i*100+j;
      h_g_data[index*3+0] = (GLubyte)0;
      h_g_data[index*3+1] = (GLubyte)0;
      h_g_data[index*3+2] = (GLubyte)0;
    }
}

void PMLInit()
{
  float Z;

  //PML init
  for(int i=0;i<(int)GRID_SIZE.y;i++){
    for(int j=0;j<(int)GRID_SIZE.x;j++){
      Z = (ECX[i] * delta_t)/(2.0*epsilon_M[i][j]);
      CEZX[i][j]=(1-Z)/(1+Z);
      CEZXL[i][j]=(delta_t/epsilon_M[i][j])/(1+Z)*(1.0/delta_x);
      CHYX[i][j]=(1-Z)/(1+Z);
      CHYXL[i][j]=(delta_t/mu_M[i][j])*(1.0/delta_x);
      Z = (ECY[j]*delta_t)/(2.0*epsilon_M[i][j]);
      CEZY[i][j]=(1-Z)/(1+Z);
      CEZYL[i][j]=(delta_t/epsilon_M[i][j])/(1+Z)*(1.0/delta_y);
      CHXY[i][j]=(1-Z)/(1+Z);
      CHXYL[i][j]=(delta_t/mu_M[i][j])*(1.0/delta_y);
    }
  }
}

void FDTDInit()
{
  Ez_max=0.0;
  Ez_min=0.0;
  // sampling_list=(float *)malloc(sizeof(float)*100);
  // for(i=0;i<100;i++)
  // sampling_list[i]=0.0;
  float ZZ;
  for(int j = 0; j<(int)GRID_SIZE.y; j++){
    for(int i = 0; i<(int)GRID_SIZE.x; i++){
      mu_M[i][j]  =  mu0;
      epsilon_M[i][j] = epsilon0;
      sigma_M[i][j] = sigma0;
    }
  }

  // for(j = 0; j<(int)GRID_SIZE.y; j++){
  //   for(i = 0;i<(int)GRID_SIZE.x; i++){
  //     Ez[i][j] = 0.0;
  //     Hx[i][j] = 0.0;
  //     Hy[i][j] = 0.0;
  //     CEZX[i][j] = 0.0;
  //     CEZXL[i][j] = 0.0;
  //     CHYX[i][j] = 0.0;
  //     CHYXL[i][j] = 0.0;
  //     CEZY[i][j] = 0.0;
  //     CEZYL[i][j] = 0.0;
  //     CHXY[i][j] = 0.0;
  //     CHXYL[i][j] = 0.0;
  //
  //     CEZ[i][j] = 0.0;
  //     CEZLX[i][j]=0.0;
  //     CEZLY[i][j]=0.0;
  //     CHXLY[i][j]=0.0;
  //     CHYLX[i][j]=0.0;
  //   }
  // }
  //
  // for(i=0;i<(int)GRID_SIZE.x;i++){
  //   ECX[i]=0.0;
  // }
  //
  // for(j=0;j<(int)GRID_SIZE.x;j++){
  //   ECY[j]=0.0;
  // }

  for(int i=0;i<L;i++){
    ECX[i] = ecmax * pow((L-i+0.5)/L,M);
    ECX[(int)GRID_SIZE.x-i-1] = ECX[i];
    ECY[i] = ECX[i];
    ECY[(int)GRID_SIZE.y-i-1] = ECX[i];
  }

  //FDTD init
  for(int i=0;i<(int)GRID_SIZE.x;i++){
    for(int j=0;j<(int)GRID_SIZE.y;j++){
      ZZ = (sigma_M[i][j] * delta_t)/(2.0*epsilon_M[i][j]);
      CEZ[i][j]=(1-ZZ)/(1+ZZ);
      CEZLX[i][j]=(delta_t/epsilon_M[i][j])/(1+ZZ)*(1.0/delta_x);
      CEZLY[i][j]=(delta_t/epsilon_M[i][j])/(1+ZZ)*(1.0/delta_y);
      CHXLY[i][j]=delta_t/mu_M[i][j]*(1.0/delta_y);
      CHYLX[i][j]=delta_t/mu_M[i][j]*(1.0/delta_x);
    }
  }
}

// void PrintInfo()
// {
//   std::cout << "\n-------------------------------\n" << std::endl;
//
//   std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
//   std::cout << "GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
//   std::cout << "Vendor: " << glGetString(GL_VENDOR) << std::endl;
//   std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;
//
//   std::cout << "\n-------------------------------\n" << std::endl;
//
//   std::cout << "lambda: " << lambda << std::endl;
//   std::cout << "freq: " << freq << std::endl;
//   std::cout << "rectD: " << rectD << std::endl;
//   std::cout << "delta_x: " << delta_x << std::endl;
//   std::cout << "delta_y: " << delta_y << std::endl;
//   std::cout << "delta_t: " << delta_t << std::endl;
//
//   std::cout << "\n-------------------------------\n" << std::endl;
// }

void ParamInit()
{
  lambda = c / freq;

  power_x=10;
  power_y=(int)GRID_SIZE.y/2-1;


  delta_x = lambda / resolution;
  delta_y = lambda / resolution;
  delta_t = (1.0 / (sqrt(pow((1 / delta_x), 2.0)+pow((1 / delta_y), 2.0))))*(1.0 / c)*alpha;

  // rectD=lambda/2/delta_x;
  rectD=lambda/2/delta_x + 6;

  anim_dt = (1.0 / (sqrt(pow((1 / delta_x), 2.0)+pow((1 / delta_y), 2.0))))*(1.0 / c)*alpha;
  step = 1.0 / freq / delta_t;
  mu0 = 1.0e-7f * 4.0 * M_PI;
  ecmax = -(M+1)*epsilon0*c / (2.0*L*delta_x)*r0;
  Ez_range = fabs(Ez_max)+fabs(Ez_min); // 2.7800852e-03f

  Ez_yellow = Ez_range*0.6f+Ez_min;
  Ez_green = Ez_range*0.50f+Ez_min;
  Ez_lightblue = Ez_range*0.4f+Ez_min;

}

void AllocInit()
{
  Ez  =  (float **)malloc(sizeof(float*) * (int)GRID_SIZE.y);
  Hx  =  (float **)malloc(sizeof(float*) * (int)GRID_SIZE.y);
  Hy  =  (float **)malloc(sizeof(float*) * (int)GRID_SIZE.y);

  sigma_M  =  (float **)malloc(sizeof(float*) * (int)GRID_SIZE.y);
  epsilon_M =  (float **)malloc(sizeof(float*) * (int)GRID_SIZE.y);
  mu_M    =  (float **)malloc(sizeof(float*) * (int)GRID_SIZE.y);

  ECX = (float *)malloc(sizeof(float) * (int)GRID_SIZE.x);
  memset(ECX, 0, sizeof(float)*(int)GRID_SIZE.x);
  ECY = (float *)malloc(sizeof(float) * (int)GRID_SIZE.y);
  memset(ECY, 0, sizeof(float)*(int)GRID_SIZE.x);

  CEZX  = (float **)malloc(sizeof(float*) * (int)GRID_SIZE.y);
  CEZXL = (float **)malloc(sizeof(float*) * (int)GRID_SIZE.y);
  CHYX  = (float **)malloc(sizeof(float*) * (int)GRID_SIZE.y);
  CHYXL = (float **)malloc(sizeof(float*) * (int)GRID_SIZE.y);
  CEZY  = (float **)malloc(sizeof(float*) * (int)GRID_SIZE.y);
  CEZYL = (float **)malloc(sizeof(float*) * (int)GRID_SIZE.y);
  CHXY  = (float **)malloc(sizeof(float*) * (int)GRID_SIZE.y);
  CHXYL = (float **)malloc(sizeof(float*) * (int)GRID_SIZE.y);

  EZX = (float **)malloc(sizeof(float*) * (int)GRID_SIZE.y);
  EZY  = (float **)malloc(sizeof(float*) * (int)GRID_SIZE.y);
  HXY = (float **)malloc(sizeof(float*) * (int)GRID_SIZE.y);
  HYX = (float **)malloc(sizeof(float*) * (int)GRID_SIZE.y);


  CEZ  = (float **)malloc(sizeof(float*) * (int)GRID_SIZE.y);
  CEZLX = (float **)malloc(sizeof(float*) * (int)GRID_SIZE.y);
  CEZLY  = (float **)malloc(sizeof(float*) * (int)GRID_SIZE.y);
  CHXLY = (float **)malloc(sizeof(float*) * (int)GRID_SIZE.y);
  CHYLX  = (float **)malloc(sizeof(float*) * (int)GRID_SIZE.y);

  for(int i=0;i<(int)GRID_SIZE.y;i++){
    Ez[i] = (float *)malloc(sizeof(float) * (int)GRID_SIZE.x);
    memset(Ez[i], 0, sizeof(float)*(int)GRID_SIZE.x);
    Hx[i] = (float *)malloc(sizeof(float) * (int)GRID_SIZE.x);
    memset(Hx[i], 0, sizeof(float)*(int)GRID_SIZE.x);
    Hy[i] = (float *)malloc(sizeof(float) * (int)GRID_SIZE.x);
    memset(Hy[i], 0, sizeof(float)*(int)GRID_SIZE.x);
    sigma_M[i] = (float *)malloc(sizeof(float) * (int)GRID_SIZE.x);
    memset(sigma_M[i], 0, sizeof(float)*(int)GRID_SIZE.x);
    epsilon_M[i] = (float *)malloc(sizeof(float) * (int)GRID_SIZE.x);
    memset(epsilon_M[i], 0, sizeof(float)*(int)GRID_SIZE.x);
    mu_M[i] = (float *)malloc(sizeof(float) * (int)GRID_SIZE.x);
    memset(mu_M[i], 0, sizeof(float)*(int)GRID_SIZE.x);

    CEZX[i]  = (float *)malloc(sizeof(float) * (int)GRID_SIZE.x);
    memset(CEZX[i], 0, sizeof(float)*(int)GRID_SIZE.x);
    CEZXL[i] = (float *)malloc(sizeof(float) * (int)GRID_SIZE.x);
    memset(CEZXL[i], 0, sizeof(float)*(int)GRID_SIZE.x);
    CHYX[i]  = (float *)malloc(sizeof(float) * (int)GRID_SIZE.x);
    memset(CHYX[i], 0, sizeof(float)*(int)GRID_SIZE.x);
    CHYXL[i] = (float *)malloc(sizeof(float) * (int)GRID_SIZE.x);
    memset(CHYXL[i], 0, sizeof(float)*(int)GRID_SIZE.x);
    CEZY[i]  = (float *)malloc(sizeof(float) * (int)GRID_SIZE.x);
    memset(CEZY[i], 0, sizeof(float)*(int)GRID_SIZE.x);
    CEZYL[i] = (float *)malloc(sizeof(float) * (int)GRID_SIZE.x);
    memset(CEZYL[i], 0, sizeof(float)*(int)GRID_SIZE.x);
    CHXY[i]  = (float *)malloc(sizeof(float) * (int)GRID_SIZE.x);
    memset(CHXY[i], 0, sizeof(float)*(int)GRID_SIZE.x);
    CHXYL[i] = (float *)malloc(sizeof(float) * (int)GRID_SIZE.x);
    memset(CHXYL[i], 0, sizeof(float)*(int)GRID_SIZE.x);

    EZX[i] = (float *)malloc(sizeof(float) * (int)GRID_SIZE.x);
    memset(EZX[i], 0, sizeof(float)*(int)GRID_SIZE.x);
    EZY[i] = (float *)malloc(sizeof(float) * (int)GRID_SIZE.x);
    memset(EZY[i], 0, sizeof(float)*(int)GRID_SIZE.x);
    HXY[i] = (float *)malloc(sizeof(float) * (int)GRID_SIZE.x);
    memset(HXY[i], 0, sizeof(float)*(int)GRID_SIZE.x);
    HYX[i] = (float *)malloc(sizeof(float) * (int)GRID_SIZE.x);
    memset(HYX[i], 0, sizeof(float)*(int)GRID_SIZE.x);

    CEZ[i] =  (float *)malloc(sizeof(float) * (int)GRID_SIZE.x);
    memset(CEZ[i], 0, sizeof(float)*(int)GRID_SIZE.x);
    CEZLX[i] = (float *)malloc(sizeof(float) * (int)GRID_SIZE.x);
    memset(CEZLX[i], 0, sizeof(float)*(int)GRID_SIZE.x);
    CEZLY[i] = (float *)malloc(sizeof(float) * (int)GRID_SIZE.x);
    memset(CEZLY[i], 0, sizeof(float)*(int)GRID_SIZE.x);
    CHXLY[i] = (float *)malloc(sizeof(float) * (int)GRID_SIZE.x);
    memset(CHXLY[i], 0, sizeof(float)*(int)GRID_SIZE.x);
    CHYLX[i] = (float *)malloc(sizeof(float) * (int)GRID_SIZE.x);
    memset(CHYLX[i], 0, sizeof(float)*(int)GRID_SIZE.x);
  }

}
void AllocFree()
{
  for(int i=0;i<(int)GRID_SIZE.y;i++)
  {
    free(Ez[i]);
    free(Hx[i]);

    free(sigma_M[i]);
    free(epsilon_M[i]);
    free(mu_M[i]);

    free(CEZX[i]);
    free(CEZXL[i]);
    free(CHYX[i]);
    free(CHYXL[i]);
    free(CEZY[i]);
    free(CEZYL[i]);
    free(CHXY[i]);
    free(CHXYL[i]);

    free(EZX[i]);
    free(EZY[i]);
    free(HXY[i]);
    free(HYX[i]);

    free(CEZ[i]);
    free(CEZLX[i]);
    free(CEZLY[i]);
    free(CHXLY[i]);
    free(CHYLX[i]);
  }
  free(Ez);
  free(Hx);
  free(Hy);

  free(sigma_M);
  free(epsilon_M);
  free(mu_M);

  free(ECX);
  free(ECY);

  free(CEZX);
  free(CEZXL);
  free(CHYX);
  free(CHYXL);
  free(CEZY);
  free(CEZYL);
  free(CHXY);
  free(CHXYL);

  free(EZX);
  free(EZY);
  free(HXY);
  free(HYX);

  free(CEZ);
  free(CEZLX);
  free(CEZLY);
  free(CHXLY);
  free(CHYLX);
}

void LoadShaders()
{
  std::vector<tdogl::Shader> shaders;
  shaders.push_back(tdogl::Shader::shaderFromFile("./vertex-shader.glsl", GL_VERTEX_SHADER));
  shaders.push_back(tdogl::Shader::shaderFromFile("./fragment-shader.glsl", GL_FRAGMENT_SHADER));
  gProgram = new tdogl::Program(shaders);
}

void LoadTriangle()
{
  glGenVertexArrays(1, &gVAO);
  glBindVertexArray(gVAO);

  glGenBuffers(1, &gVBO);
  glBindBuffer(GL_ARRAY_BUFFER, gVBO);

  GLfloat vertexData[] =
  {
    1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
    1.0f, 1.0f, 0.0f, 1.0f, 1.0f,
    -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,

    -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
    1.0f, 1.0f, 0.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, 0.0f, 0.0f, 1.0f
  };
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertexData), vertexData, GL_STATIC_DRAW);

  glEnableVertexAttribArray(gProgram->attrib("vert"));
  glVertexAttribPointer(gProgram->attrib("vert"), 3, GL_FLOAT, GL_FALSE, 5*sizeof(GLfloat), NULL);

  glEnableVertexAttribArray(gProgram->attrib("vertTexCoord"));
  glVertexAttribPointer(gProgram->attrib("vertTexCoord"), 2, GL_FLOAT, GL_FALSE, 5*sizeof(GLfloat), (const GLvoid*)(3 * sizeof(GLfloat)));


  glBindVertexArray(0);
}

void Render()
{

  glClearColor(0, 0, 0, 1);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  gProgram->use();

  gProgram->setUniform("camera", gCamera.matrix());

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, tex);
  gProgram->setUniform("tex", 0);

  glBindVertexArray(gVAO);

  glDrawArrays(GL_TRIANGLES, 0, 6);

  glBindVertexArray(0);
  glBindTexture(GL_TEXTURE_2D, 0);
  gProgram->stopUsing();

}

void OnError(int errorCode, const char *msg)
{
  throw std::runtime_error(msg);
}
void AppMain()
{
  GLFW_GLEWInit();

  LoadShaders();

  AllocInit();
  atexit(AllocFree);

  ParamInit();

  FDTDInit();

  PMLInit();

  // PrintInfo();

  TextureInit();

  CreateTexture(1, &tex);

  LoadTriangle();

  CameraInit();

  MouseInit();


  ctx = nk_glfw3_init(gWindow, NK_GLFW3_INSTALL_CALLBACKS);
  {
    struct nk_font_atlas *atlas;
    nk_glfw3_font_stash_begin(&atlas);
    nk_glfw3_font_stash_end();
  }


  glfwSwapInterval(1);

  float lastTime = glfwGetTime();
  while(!glfwWindowShouldClose(gWindow))
  {
    glfwPollEvents();

    nk_glfw3_new_frame();

    RunCPUKernel();

    Render();

    // CalcFPS(gWindow, 1.0, "GL3.2 FDTD2D_TM");

    float thisTime = glfwGetTime();
    Update((float)(thisTime - lastTime));
    lastTime = thisTime;

    GUIRender(ctx);

    nk_glfw3_render(NK_ANTI_ALIASING_ON, MAX_VERTEX_BUFFER, MAX_ELEMENT_BUFFER);

    glfwSwapBuffers(gWindow);


    if(glfwGetKey(gWindow, GLFW_KEY_ESCAPE))
    {
      free(h_g_data);
      DeleteTexture(1, &tex);
      glfwSetWindowShouldClose(gWindow, GL_TRUE);
    }
  }
  nk_glfw3_shutdown();
  glfwTerminate();
}

int main(int argc, char const* argv[])
{
  try {
    AppMain();
  } catch (const std::exception &e) {
    std::cerr << "ERRPR : " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
