all:
	g++ -c Shader.cpp
	g++ -c Program.cpp
	g++ -c Camera.cpp
	nvcc -c Kernel.cu
	g++ -c main.cpp -I./nuklear -I./glm -lglfw3 -lGLEW -lGL -lXrandr -lXinerama -lXcursor -lXi -lXxf86vm -lX11 -lpthread -lrt -lm -std=c++11 -Wall 
	nvcc -o fdtd Kernel.o Program.o Shader.o Camera.o main.o -Xcompiler "-lglfw3 -lGLEW -lGL -lXrandr -lXinerama -lXcursor -lXi -lXxf86vm -lX11 -lpthread -lrt -lm -std=c++11 -fno-use-linker-plugin" -Xptxas -v -lglfw3 
