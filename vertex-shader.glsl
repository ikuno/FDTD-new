#version 150

uniform mat4 camera;

in vec3 vert;
in vec2 verTexCoord;

out vec2 fragTexCoord;

void main(){
  fragTexCoord = verTexCoord;

  gl_Position = camera * vec4(vert, 1);
}
