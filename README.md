#FDTD + OpenGL + CUDA/OpenMP

## Explanation
- 2D Microwave simulation by using FDTD TM mode.
- RealTime Visualization by OpenGL.
- Parallelization wit CUDA/OpenMP

## Request
- GCC >= 4.9
- GLFW
- GLEW 
- OpenGL >= 3.2
- GLSL >= 1.5
- CUDA (Unified Memory need CUDA Toolkit >= 6, Capability >= 3)
- OpenMP

## Install
- git clone --recursive -j8 https://github.com/Jie211/fdtd-tmp.git
- or
    - git clone https://github.com/Jie211/fdtd-tmp.git
    - git submodule update --init --recursive
