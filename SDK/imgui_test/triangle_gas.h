#ifndef __TRIANGLE_GAS_H__
#define __TRIANGLE_GAS_H__

#include <spdlog/spdlog.h>
#include <string>
#include <tuple>

#include "cuda_runtime_api.h"
#include "device.h"
#include "driver_types.h"
#include "optix_types.h"
#include "vector_types.h"

#include <imgui/imgui.h>

class TriangleGAS
{
  public:
    TriangleGAS(const Device &device, const std::string &filename);

    TriangleGAS(const TriangleGAS &other)    = delete;
    TriangleGAS(TriangleGAS &&other)         = delete;
    void operator=(const TriangleGAS &other) = delete;
    void operator=(TriangleGAS &&other)      = delete;


    ~TriangleGAS();

    [[nodiscard]] OptixTraversableHandle get_gas_handle() const noexcept { return m_gas_handle; }

    [[nodiscard]] float3 *get_device_normals() const noexcept
    {
        spdlog::info("copy {} normals to device", m_normals.size());
        float3 *ptr = nullptr;
        cudaMalloc(reinterpret_cast<void **>(&ptr), sizeof(float3) * m_normals.size());
        cudaMemcpy(reinterpret_cast<void *>(ptr),
            reinterpret_cast<const void *>(m_normals.data()),
            sizeof(float3) * m_normals.size(),
            cudaMemcpyHostToDevice);
        return ptr;
    }

    [[nodiscard]] float3 *get_device_vertices() const noexcept
    {
        spdlog::info("copy {} vertices to device", m_vertices.size());
        float3 *ptr = nullptr;
        cudaMalloc(reinterpret_cast<void **>(&ptr), sizeof(float3) * m_vertices.size());
        cudaMemcpy(reinterpret_cast<void *>(ptr),
            reinterpret_cast<const void *>(m_vertices.data()),
            sizeof(float3) * m_vertices.size(),
            cudaMemcpyHostToDevice);
        return ptr;
    }

    [[nodiscard]] std::vector<int> const &get_mat_indices() const { return m_mat_indices; }

    void imgui() const
    {
        if (ImGui::CollapsingHeader("Triangle GAS")) {
            ImGui::Text("vertices: %d K", m_vertices.size() / 1000);
            ImGui::Text("GAS buffer size: %d MB", m_gas_buffer_sizes.outputSizeInBytes / 1024 / 1024);
        }
    }

  private:
    OptixTraversableHandle m_gas_handle;
    OptixAccelBufferSizes  m_gas_buffer_sizes;
    CUdeviceptr            m_d_gas_output_buffer;
    std::vector<float3>    m_vertices;
    std::vector<float3>    m_normals;
    std::vector<int>       m_mat_indices;
};

std::tuple<std::vector<float3>, std::vector<float3>, std::vector<int>> load_assimp(const std::string &filename);

#endif
