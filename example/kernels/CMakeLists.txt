get_filename_component(CUDA_COMPILER_BIN "${CMAKE_CUDA_COMPILER}" DIRECTORY)
get_filename_component(CUDA_DIR "${CUDA_COMPILER_BIN}" DIRECTORY)
set(CUDA_INCLUDE "${CUDA_DIR}/include")

set(PROJ_NAME "example_kernels")
set(SOURCE
    CubeRotator.cu
    ImageProcessor.cu
    ../tools/CudaHelper.cpp
)
add_library(${PROJ_NAME} STATIC)
target_sources(${PROJ_NAME}
    PRIVATE
        ${SOURCE}
)
target_include_directories(${PROJ_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_DIR}
        ${CUDA_INCLUDE}
)
target_link_libraries(${PROJ_NAME} 
    PRIVATE
        gl_render
)
set_target_properties(${PROJ_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJ_NAME} PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH})