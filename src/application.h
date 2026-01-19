//
// Created by james on 10/01/2026.
//
#pragma once

#define GLFW_INCLUDE_VULKAN
#include <optional>
#include <string>
#include <vector>
#include <GLFW/glfw3.h>
#include <vulkan/vulkan_beta.h>

class application {
public:
    application();
    ~application();

    void run();

private:
    void checkValidationLayerSupport();
    void cleanup();
    void cleanupSwapChain();
    void createCommandBuffer();
    void createCommandPool();
    void createDescriptorPool();
    void createDescriptorSets();
    void createDescriptorSetLayout();
    void createFrameBuffers();
    void createGraphicsPipeline();
    void createImageViews();
    void createIndexBuffer();
    void createInstance();
    void createLogicalDevice();
    void createRenderPass();
    void createSurface();
    void createSwapChain();
    void createSyncObjects();
    void createUniformBuffers();
    void createVertexBuffer();
    void drawFrame();
    void initWindow();
    void initVulkan();
    void mainLoop();
    void pickPhysicalDevice();
    void recreateSwapChain();
    void setupDebugMessenger();
    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
    void createTextureImage();

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);

    struct QueueFamilyIndices {
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;

        bool isComplete() {
            return graphicsFamily.has_value() && presentFamily.has_value();
        }
    };

    struct SwapChainSupportDetails {
        VkSurfaceCapabilitiesKHR capabilities;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
    };

    VkPresentModeKHR chooseSwapChainPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    void copyBuffer(VkBuffer dstBuffer, VkBuffer srcBuffer, VkDeviceSize size);
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
    VkShaderModule createShaderModule(const std::vector<char>& code);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    QueueFamilyIndices findQueueFamilies(const VkPhysicalDevice& device);
    SwapChainSupportDetails querySwapChainSupport(const VkPhysicalDevice& device);
    std::vector<const char*> getRequiredInstanceExtensions() const;
    static bool checkDeviceExtensionSupport(const VkPhysicalDevice& device);
    bool isDeviceSuitable(const VkPhysicalDevice& device) ;
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);
    void updateUniformBuffer(uint32_t currentImage);

    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer commandBuffer);

    GLFWwindow* m_pWindow;

    std::vector<VkCommandBuffer> m_commandBuffers;
    VkBuffer m_indexBuffer;
    VkDescriptorPool m_descriptorPool;
    std::vector<VkDescriptorSet> m_descriptorSets;
    VkDeviceMemory m_indexBufferMemory;
    VkBuffer m_stagingBuffer;
    VkDeviceMemory m_stagingBufferMemory;
    VkBuffer m_vertexBuffer;
    VkDeviceMemory m_vertexBufferDeviceMemory;
    VkCommandPool m_commandPool;
    VkInstance m_instance;
    VkDebugUtilsMessengerEXT m_debugMessenger;
    VkDevice m_device;
    VkPhysicalDevice m_physicalDevice;
    VkPipeline m_graphicsPipeline;
    VkDescriptorSetLayout m_descriptorSetLayout;
    VkPipelineLayout m_pipelineLayout;
    VkQueue m_graphicsQueue;
    VkQueue m_presentQueue;
    VkRenderPass m_renderPass;
    VkSurfaceKHR m_surface;
    VkSwapchainKHR m_swapChain;
    std::vector<VkFramebuffer> m_swapChainFramebuffers;
    std::vector<VkImage> m_swapChainImages;
    std::vector<VkImageView> m_swapChainImageViews;
    VkImage m_textureImage;
    VkDeviceMemory m_textureImageMemory;
    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;
    VkFormat m_swapChainImageFormat;
    VkExtent2D m_swapChainExtent;

    std::vector<VkSemaphore> m_imageAvailableSemaphores;
    std::vector<VkSemaphore> m_renderFinishedSemaphores;
    std::vector<VkFence> m_inFlightFences;

    bool framebufferResized = false;

    uint32_t m_currentFrame = 0;

    static void populateDebugMessengerInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData);
    static std::vector<char> readFile(const std::string& filename);

    static const uint32_t s_windowWidth = 800;
    static const uint32_t s_windowHeight = 600;

    static const uint32_t s_maxFramesInFlight = 2;

    static const std::vector<const char*> deviceExtensions;
    static const std::vector<const char*> validationLayers;

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif
};
