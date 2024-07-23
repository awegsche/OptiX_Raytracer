#include "device.h"

#include <imgui/imgui.h>
#include <optix_types.h>
#include <spdlog/spdlog.h>
#include <sutil/Exception.h>

#include <optix_stubs.h>

static void context_log_cb(unsigned int level, const char *tag, const char *message, void * /*cbdata */)
{
    spdlog::info("{:3} | {:12} | {}", level, message, tag);
}

Device::Device()
{
    //
    spdlog::info("Initialize CUDA and create OptiX context");
    //
    // Initialize CUDA
    CUDA_CHECK(cudaFree(nullptr));

    CUDA_CHECK(cudaDriverGetVersion(&m_driver_version));
    CUDA_CHECK(cudaRuntimeGetVersion(&m_runtime_version));

    spdlog::info(" CUDA driver version: {}", m_driver_version);
    spdlog::info(" CUDA runtim version: {}", m_runtime_version);

    // Initialize the OptiX API, loading all API entry points
    OPTIX_CHECK(optixInit());

    // Specify context options
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
#ifdef DEBUG
    // This may incur significant performance cost and should only be done during development.
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif

    // Associate a CUDA context (and therefore a specific GPU) with this
    // device context
    OPTIX_CHECK(optixDeviceContextCreate(m_cu_ctx, &options, &m_context));
}

void Device::imgui() const noexcept
{
    if (ImGui::CollapsingHeader("Device")) {
        ImGui::Text("CUDA driver version: %d.%d",
            m_driver_version / 1000,
            m_driver_version / 10 - m_driver_version / 1000 * 100);
        ImGui::Text("CUDA runtime version: %d.%d",
            m_runtime_version / 1000,
            m_runtime_version / 10 - m_runtime_version / 1000 * 100);
    }
}
