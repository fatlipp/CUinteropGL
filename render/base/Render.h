#pragma once

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

#include <functional>

class Render
{
public:
    Render(const int width, const int height);
    ~Render();

public:
    void SetShader(GLuint shader)
    {
        shaderProgram = shader;    
    }

    void Initialize();
    void StartLoop();

    void SetDrawCallback(std::function<void()> cb)
    {
        onDraw = cb;
    }

private:
    const int width;
    const int height;

    GLFWwindow* window;
    GLuint shaderProgram;
    glm::vec3 position = glm::vec3( 0, 0.0, -5 );
    glm::mat4 modelMatrix = glm::mat4(1.0);
    glm::mat4 viewMatrix = glm::mat4(1.0);
    glm::mat4 projectionMatrix = glm::mat4(1.0);

    std::function<void()> onDraw;
};