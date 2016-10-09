CUDA_INSTALL_PATH ?= /usr/local/cuda

CXX := g++
NVCC := nvcc
LINK := nvcc

INCLUDES = -I./nuklear -I./glm

NVCCINFO = -Xptxas -v
NVCCFLAGS = -std=c++11 -O3 --use_fast_math
# NVCCFLAGS += $(NVCCINFO)

CXXFLAGS = -O3

LINKFLAGS = -Xcompiler "-lglfw3 -lGLEW -lGL -lXrandr -lXinerama -lXcursor -lXi -lXxf86vm -lX11 -lpthread -lrt -lm -std=c++11 -fno-use-linker-plugin" -lglfw3


OBJS = Camera.o Program.o Shader.o main.o 
TARGET = fdtd

.PHONY: clean all 

all: $(TARGET)

clean:
	$(RM) $(OBJS) $(TARGET)

LINKLINE = $(LINK) -o $(TARGET) $(LINKFLAGS) $(OBJS)

.SUFFIXES: .cpp .cu .o

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(TARGET): $(OBJS) Makefile
	$(LINKLINE)
