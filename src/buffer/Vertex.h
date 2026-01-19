//
// Created by James on 19/01/2026.
//

#pragma once

#include <vulkan/vulkan.h>

struct Vertex {
    glm::vec2 position;
    glm::vec3 color;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDesc{};

        bindingDesc.binding = 0;
        bindingDesc.stride = sizeof(Vertex);
        bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDesc;
    }

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attriDescs{};

        attriDescs[0].binding = 0;
        attriDescs[0].location = 0;
        attriDescs[0].format = VK_FORMAT_R32G32_SFLOAT;
        attriDescs[0].offset = offsetof(Vertex, position);

        attriDescs[1].binding = 0;
        attriDescs[1].location = 1;
        attriDescs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attriDescs[1].offset = offsetof(Vertex, color);

        return attriDescs;
    }
};