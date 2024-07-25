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

class TriangleGAS
{
  public:
    TriangleGAS(const Device &device, const std::string &filename);

    TriangleGAS(const TriangleGAS &other) = delete;
    TriangleGAS(TriangleGAS &&other) = delete;
    void operator=(const TriangleGAS &other) = delete;
    void operator=(TriangleGAS &&other) = delete;


    ~TriangleGAS();

    [[nodiscard]] OptixTraversableHandle get_gas_handle() const noexcept { return m_gas_handle; }

    [[nodiscard]] float3* get_device_normals() const noexcept {
        float3* ptr = nullptr;
        cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(float3) * m_normals.size());
        cudaMemcpy(reinterpret_cast<void*>(ptr), reinterpret_cast<const void*>(m_normals.data()), sizeof(float3) * m_normals.size(), cudaMemcpyHostToDevice);
        return ptr;
    }

    [[nodiscard]] float3* get_device_vertices() const noexcept {
        spdlog::info("copy {} vertices to device", m_vertices.size());
        float3* ptr = nullptr;
        cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(float3) * m_vertices.size());
        cudaMemcpy(reinterpret_cast<void*>(ptr), reinterpret_cast<const void*>(m_vertices.data()), sizeof(float3) * m_vertices.size(), cudaMemcpyHostToDevice);
        return ptr;
    }

  private:
    OptixTraversableHandle m_gas_handle;
    CUdeviceptr m_d_gas_output_buffer;
    std::vector<float3> m_vertices;
    std::vector<float3> m_normals;
};

std::tuple<std::vector<float3>, std::vector<float3>> load_assimp(const std::string &filename);

#endif
