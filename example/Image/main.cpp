#include "Shader2d.h"

#include "render/base/Render.h"
#include "example/kernels/ImageProcessor.cuh"
#include <tuple>
#include <iostream>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

std::tuple<GLuint, GLuint, GLuint, GLuint> CreateTextureBo(float cubeScale, float aspectRatio)
{
    if (cubeScale < 0.1f)
    {
        cubeScale = 0.1f;
    }

    GLuint vao;
    GLuint vbo;
    GLuint vboTex;
    GLuint ebo;

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &vboTex);
    glGenBuffers(1, &ebo);

    struct float2 {float x; float y; };

    const float minX = -cubeScale * aspectRatio;
    const float minY = -cubeScale;
    const float maxX = cubeScale * aspectRatio;
    const float maxY = cubeScale;

    float2 vertices[4];
    vertices[0] = {maxX, maxY};
    vertices[1] = {maxX, minY};
    vertices[2] = {minX, minY};
    vertices[3] = {minX, maxY};

    float2 textCoords[4];
    textCoords[0] = {1.0f, 1.0f};
    textCoords[1] = {1.0f, 0.0f};
    textCoords[2] = {0.0f, 0.0f};
    textCoords[3] = {0.0f, 1.0f};
    
    const unsigned indices[] = {
        0, 1, 3,
        1, 2, 3
    };

    glBindVertexArray(vao);
        // ids (6)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // Vertex attribute 0 - positions (layout = 0 in shader)
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float2) * 4, vertices, GL_STATIC_DRAW);
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float2), 0);
            glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // Vertex attribute 1 - tex coords (layout = 1 in shader)
        glBindBuffer(GL_ARRAY_BUFFER, vboTex);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float2) * 4, textCoords, GL_STATIC_DRAW);
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(float2), 0);
            glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    return {vao, vbo, vboTex, ebo};
}

std::tuple<GLuint, GLuint> CreateTexture(int width, int height, int channels, unsigned char* image)
{
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, 
            GL_UNSIGNED_BYTE, image);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        GLuint pbo;
        glGenBuffers(1, &pbo);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * channels, image, GL_STATIC_DRAW);
        
        GLint bsize;
        glGetBufferParameteriv(GL_PIXEL_UNPACK_BUFFER, GL_BUFFER_SIZE, &bsize);

        if ((GLuint)bsize != (channels * sizeof(unsigned char) * width * height)) 
        {
            printf("Buffer object (%d) has incorrect size (%d).\n",
                    (unsigned)pbo, (unsigned)bsize);
            exit(EXIT_FAILURE);
        }
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    return {texture, pbo};
}

/// @brief applies filter (grayscale) to an image after a delay (ms)
/// @param argc 
/// @param argv 
/// @return 
int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "usage: ./example_image 'path_to_image' 'delay_to_apply_filter(ms)'\n";
        return -1;
    }

    const auto delay = argc < 3 ? 1000 : std::stoi(argv[2]);

    const auto viewSize = 1500;
    const float cubeScale = 2;

    int width;
    int height;
    int channels;
    unsigned char* image = stbi_load(argv[1], &width, &height, &channels, 0);
    
    if (!image) 
    {
        fprintf(stderr, "Failed to load image\n");
        return -1;
    }

    std::cout << "width: " << width << ", ";
    std::cout << "height: " << height << ", ";
    std::cout << "channels: " << channels << "\n";

    const auto aspectRatio = static_cast<float>(width) / static_cast<float>(height);

    Render render { viewSize * aspectRatio, viewSize };

    auto shader = CreateDefaultShader2d();
    render.SetShader(shader);

    auto [vao, vbo, vboTex, ebo] = CreateTextureBo(cubeScale, aspectRatio);
    auto [texture, pbo] = CreateTexture(width, height, channels, image);

    auto proc = ImageProcessor(width, height, pbo);

    const auto startTime = std::chrono::high_resolution_clock::now();

    render.SetDrawCallback([&vbo, &vao, &pbo, &width, &height, &texture, &proc, &startTime, &delay] {

        const auto nowTime = std::chrono::high_resolution_clock::now();

        if (std::chrono::duration_cast<std::chrono::milliseconds>(nowTime - startTime).count() > delay)
        {
            proc.Process();
        }

        glBindTexture(GL_TEXTURE_2D, texture);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, 
                GL_RGB, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glBindTexture(GL_TEXTURE_2D, 0);

        GET_GL_ERROR("Draw() ERROR");
    });
    render.StartLoop();

    stbi_image_free(image);

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &vboTex);
    glDeleteBuffers(1, &ebo);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &texture);
    glDeleteProgram(shader);

    GET_GL_ERROR("Free() ERROR");

    return 0;
}