cmake_minimum_required(VERSION 4.1)
project(vulkandemo)

set(CMAKE_CXX_STANDARD 20)

# Find Vulkan
find_package(glfw3 REQUIRED)

# Find Vulkan
find_package(Vulkan REQUIRED)

add_executable(vulkandemo
        src/application.cpp
        src/log/log.cpp
        src/main.cpp
        src/buffer/vertex.h
        src/buffer/UniformBufferObject.h
        src/stb_image.cpp
)

# Link libraries
target_link_libraries(vulkandemo
        Vulkan::Vulkan
        glfw
)

# Custom command to compile shaders
find_program(GLSLC glslc REQUIRED)

add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/vert.spv
        COMMAND ${GLSLC} ${CMAKE_SOURCE_DIR}/src/shader/shader.vert -o ${CMAKE_BINARY_DIR}/vert.spv
        DEPENDS ${CMAKE_SOURCE_DIR}/src/shader/shader.vert
        COMMENT "Compiling vertex shader"
)

add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/frag.spv
        COMMAND ${GLSLC} ${CMAKE_SOURCE_DIR}/src/shader/shader.frag -o ${CMAKE_BINARY_DIR}/frag.spv
        DEPENDS ${CMAKE_SOURCE_DIR}/src/shader/shader.frag
        COMMENT "Compiling fragment shader"
)

file(COPY ${CMAKE_SOURCE_DIR}/src/textures DESTINATION ${CMAKE_BINARY_DIR}/)

add_custom_target(shaders ALL
        DEPENDS ${CMAKE_BINARY_DIR}/vert.spv ${CMAKE_BINARY_DIR}/frag.spv
)

add_dependencies(vulkandemo shaders)