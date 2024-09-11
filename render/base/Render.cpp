#include <GL/glew.h>

#include "base/Render.h"
#include "render/tools/OpenGlHelper.h"


#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <stdio.h>
#include <iostream>

Render::Render(const int width, const int height)
    : width {width}
    , height {height}
{
    if (!glfwInit()) 
    {
        fprintf(stderr, "Failed to GLFW init\n");
        exit(-1);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(width, height, "Render", NULL, NULL);
    if (!window) 
    {
        fprintf(stderr, "Failed to create a Window\n");
        glfwTerminate();
        exit(-1);
    }

    glfwMakeContextCurrent(window);

    if (glewInit() != GLEW_OK) 
    {
        fprintf(stderr, "Failed to initialize GLEW\n");
        exit(-1);
    }
}

void Render::StartLoop()
{
    std::cout << "StartLoop\n" << "\n";
    while (!glfwWindowShouldClose(window)) 
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        const glm::vec3 direction(
                cos(0) * sin(0),
                sin(0),
                cos(0) * cos(0)
            );
        const glm::vec3 right = glm::vec3(
                sin(0 - 3.14f / 2.0f),
                0,
                cos(0 - 3.14f / 2.0f)
            );
        const glm::vec3 up = glm::cross(right, direction );

        projectionMatrix = glm::perspective(glm::radians(60.0f), static_cast<float>(width) / height, 0.01f, 10000.0f);
        viewMatrix = glm::lookAt(position, position + direction, up);

        glUseProgram(shaderProgram);
        
        const GLuint modelLoc = glGetUniformLocation(shaderProgram, "model");
        const GLuint viewLoc = glGetUniformLocation(shaderProgram, "view");
        const GLuint projectionLoc = glGetUniformLocation(shaderProgram, "projection");

        GET_GL_ERROR("Pre Draw error\n");

        if (onDraw)
        {
            onDraw();
        }

        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(modelMatrix));
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(viewMatrix));
        glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projectionMatrix));

        glfwSwapBuffers(window);
        glfwPollEvents();

        glUseProgram(0);

        GET_GL_ERROR("Loop() ERROR\n");
    }
}

Render::~Render()
{
    glfwTerminate();
}