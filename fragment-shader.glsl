#version 150

uniform sampler2D tex;

in vec2 fragTexCoord;

out vec4 finalColor;

void main(){
  /* finalColor = vec4(1.0, 1.0, 1.0, 1.0); */
  finalColor = texture(tex, fragTexCoord);
}
