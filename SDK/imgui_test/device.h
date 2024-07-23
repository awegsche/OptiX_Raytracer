#include <cuda.h>
#include <cuda_runtime_api.h>
#include <optix.h>


class Device
{
  public:
    Device();

    [[nodiscard]] OptixDeviceContext get_context() const noexcept { return m_context; }

    void imgui() const noexcept;

  private:
    OptixDeviceContext m_context{ nullptr };
    CUcontext m_cu_ctx{ nullptr };
    int m_driver_version{ 0 };
    int m_runtime_version{ 0 };
};
