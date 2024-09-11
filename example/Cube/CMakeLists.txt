get_filename_component(CUDA_COMPILER_BIN "${CMAKE_CUDA_COMPILER}" DIRECTORY)
get_filename_component(CUDA_DIR "${CUDA_COMPILER_BIN}" DIRECTORY)
set(CUDA_INCLUDE "${CUDA_DIR}/include")

set(PROJ_NAME "example_cube")
set(SOURCE
    main.cpp
)
add_executable(${PROJ_NAME})
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
        example_kernels
)
set_target_properties(${PROJ_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJ_NAME} PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH})