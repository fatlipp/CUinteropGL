#include "Shader3d.h"

#include "example/kernels/CubeRotator.cuh"
#include "render/base/Render.h"
#include <tuple>
#include <iostream>

std::tuple<GLuint, GLuint, GLuint, GLuint> CreateCubeBo(float cubeScale)
{
    if (cubeScale < 0.1f)
    {
        cubeScale = 0.1f;
    }

    GLuint vbo;
    GLuint vao;
    GLuint vco;
    GLuint ebo;

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &vco);
    glGenBuffers(1, &ebo);

    struct float3 {float x; float y; float z;};

    const float minX = -cubeScale;
    const float minY = -cubeScale;
    const float minZ = -cubeScale;
    const float maxX = cubeScale;
    const float maxY = cubeScale;
    const float maxZ = cubeScale;

    float3 cube[8];
    cube[0] = {minX, minY, minZ};
    cube[1] = {minX, maxY, minZ};
    cube[2] = {maxX, maxY, minZ};
    cube[3] = {maxX, minY, minZ};
    cube[4] = {minX, minY, maxZ};
    cube[5] = {minX, maxY, maxZ};
    cube[6] = {maxX, maxY, maxZ};
    cube[7] = {maxX, minY, maxZ};
    
    float3 color[8];
    color[0] = { 1.0f, 0.0f, 0.0f };
    color[1] = { 0.0f, 1.0f, 0.0f };
    color[2] = { 0.0f, 0.0f, 1.0f };
    color[3] = { 1.0f, 1.0f, 0.0f };
    color[4] = { 0.0f, 1.0f, 1.0f };
    color[5] = { 1.0f, 0.0f, 1.0f };
    color[6] = { 1.0f, 1.0f, 1.0f };
    color[7] = { 0.5f, 1.0f, 0.5f };

    const unsigned indices[] = {
        0, 1, 2, 3, 0,
        4, 5, 6, 7, 4,
        5, 1, 2, 6, 7, 
        3
    };

    glBindVertexArray(vao);
        // ids (16)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
        // Vertex attribute 0 - positions (layout = 0 in shader)
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float3) * 8, cube, GL_STATIC_DRAW);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float3), 0);
            glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        // Vertex attribute 1 - colors (layout = 1 in shader)
        glBindBuffer(GL_ARRAY_BUFFER, vco);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float3) * 8, color, GL_STATIC_DRAW);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float3), 0);
            glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    return {vao, vbo, vco, ebo};
}

int main(int argc, char* argv[])
{
    const auto viewSize = 1500;
    Render render { viewSize, viewSize };

    auto shader = CreateDefaultShader3d();
    render.SetShader(shader);

    const float cubeScale = 1.0f;
    auto [vao, vbo, vco, ebo] = CreateCubeBo(cubeScale);

    // register the VBO object in CUDA
    CubeRotator proc{vbo, 8};

    render.SetDrawCallback([&vao, &proc]{
        proc.Process();

        glBindVertexArray(vao);
            glLineWidth(5);
            glDrawElements(GL_LINE_STRIP, 16, GL_UNSIGNED_INT, 0);
            glLineWidth(1);
        glBindVertexArray(0);

        GET_GL_ERROR("Draw() ERROR");
    });
    render.StartLoop();

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &vco);
    glDeleteBuffers(1, &ebo);
    glDeleteProgram(shader);
    
    GET_GL_ERROR("Free() ERROR");

    return 0;
}