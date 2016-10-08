all:
	g++ -c Shader.cpp
	g++ -c Program.cpp
	g++ -c Camera.cpp
	nvcc -c main.cu -I./nuklear -I./glm -lglfw3 -lGLEW -lGL -lXrandr -lXinerama -lXcursor -lXi -lXxf86vm -lX11 -lpthread -lrt -lm -std=c++11
	nvcc -o fdtd main.o Shader.o Program.o Camera.o -Xcompiler "-lglfw3 -lGLEW -lGL -lXrandr -lXinerama -lXcursor -lXi -lXxf86vm -lX11 -lpthread -lrt -lm -std=c++11 -fno-use-linker-plugin" -Xptxas -v -lglfw3

