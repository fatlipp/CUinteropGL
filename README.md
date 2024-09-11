## CUDA-OpenGL Interoperability Demo

This project demonstrates the interoperability between CUDA and OpenGL through two examples:
* Rotating an OpenGL Cube: Uses a CUDA kernel to rotate a cube by mapping its vertex buffer object (VBO) to CUDA.
* Image Grayscale Conversion: Converts an image to grayscale using a CUDA kernel by mapping a pixel buffer object (PBO) to CUDA.

**Requirements**
 * OpenGL: A cross-platform API for rendering 2D/3D graphics, essential for rendering the cube in the demo
 * GLEW: The OpenGL Extension that helps to manage OpenGL extensions and ensures access to the latest features of OpenGL
 * GLFW: A library for creating windows, managing OpenGL contexts, and handling input events, used to set up the window
 * CUDA: A parallel computing platform and API from NVIDIA, used to run the custom kernels for cube rotation and image processing

**Build**
 mkdir build && cd build
 cmake .. [-DCUDA_ARCH=...]
 make -j16

**Run**
 ./example/Cube/example_cube 
 ./example/Image/example_image 'path to image' ['delay in ms']

**Project Structure**
 - **/render**: Contains a simple rendering class. Ready to implement mouse and keyboard controls.
 - **/example**:
    - /Cube: Demonstrates rotating a cube by mapping a vertex buffer object (VBO) to CUDA and applying the rotation using a CUDA kernel.
    - /Image: Converts an image to grayscale by mapping a pixel buffer object (PBO) to CUDA and processing it using a CUDA kernel.
    - /kernels: Contains various CUDA kernels used in the examples.