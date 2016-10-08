all:
	g++ -c -O3 Shader.cpp
	g++ -c -O3 Program.cpp
	g++ -c -O3 Camera.cpp
	nvcc -c main.cu -I./nuklear -I./glm -std=c++11 -O3 --use_fast_math -Xptxas -v
	nvcc -o fdtd main.o Shader.o Program.o Camera.o -Xcompiler "-lglfw3 -lGLEW -lGL -lXrandr -lXinerama -lXcursor -lXi -lXxf86vm -lX11 -lpthread -lrt -lm -std=c++11 -fno-use-linker-plugin" -lglfw3

