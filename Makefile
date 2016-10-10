CUDA_INSTALL_PATH ?= /usr/local/cuda

CXX := g++
NVCC := $(CUDA_INSTALL_PATH)/bin/nvcc
LINK := $(CUDA_INSTALL_PATH)/bin/nvcc

INCLUDES = -I./common/inc

NVCCINFO = -Xptxas -v
# NVCCFLAGS = -Xcompiler "-fopenmp -O3" -std=c++11 -O3 --use_fast_math
NVCCFLAGS = -Xcompiler "-fopenmp " -std=c++11 --use_fast_math
# NVCCFLAGS += $(NVCCINFO)

# CXXFLAGS = -O3
CXXFLAGS = -Wall

LINKFLAGS = -Xcompiler "-fopenmp -O3 -lglfw3 -lGLEW -lGL -lXrandr -lXinerama -lXcursor -lXi -lXxf86vm -lX11 -lpthread -lrt -lm -std=c++11 -fno-use-linker-plugin" -lglfw3 -L./common/lib


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
