#ifndef __TRIANGLE_GAS_H__
#define __TRIANGLE_GAS_H__

#include <spdlog/spdlog.h>
#include <string>

#include "device.h"
#include "optix_types.h"

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

  private:
    OptixTraversableHandle m_gas_handle;
    CUdeviceptr m_d_gas_output_buffer;
};

std::vector<float3> load_assimp(const std::string &filename);

#endif
