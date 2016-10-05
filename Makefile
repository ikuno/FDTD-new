all:
	# g++ -o fdtd main.cpp Camera.cpp Program.cpp Shader.cpp Bitmap.cpp Texture.cpp -O2 -I/usr/local/include -I/usr/include -L/usr/lib -L/usr/local/lib -I./nuklear -I./glm -lglfw3 -lGLEW -framework OpenGL -Wall
	g++ -o fdtd main.cpp Camera.cpp Program.cpp Shader.cpp -I./nuklear -I./glm -lglfw3 -lGLEW -lGL -lXrandr -lXinerama -lXcursor -lXi -lXxf86vm -lX11 -lpthread -lrt -lm -std=c++0x -g  -fno-use-linker-plugin -Wall 
