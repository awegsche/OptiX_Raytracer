#pragma once

#include "optixTriangle.h"
#include <optix.h>
#include <span>
#include <sutil/Exception.h>

template<typename T> struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using TransformFloats = std::array<float, 12>;

struct Transform
{
    static Transform identity() { return { { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 } }; }

    static Transform translated(float dx, float dy, float dz) { return { { 1, 0, 0, dx, 0, 1, 0, dy, 0, 0, 1, dz } }; }

    static Transform rotated_x(float angle)
    {
        return { {
            1,
            0,
            0,
            0,
            0,
            cos(angle),
            -sin(angle),
            0,
            0,
            sin(angle),
            cos(angle),
            0,
        } };
    }
    static Transform rotated_y(float angle)
    {
        return { {
            cos(angle),
            0,
            -sin(angle),
            0,
            0,
            1,
            0,
            0,
            sin(angle),
            0,
            cos(angle),
            0,
        } };
    }
    static Transform rotated_z(float angle)
    {
        return { {
            cos(angle),
            -sin(angle),
            0,
            0,
            sin(angle),
            cos(angle),
            0,
            0,
            0,
            0,
            1,
            0,
        } };
    }

    TransformFloats m_matrix;

    Transform operator*(Transform const &other) const
    {
        Transform result;
        std::fill(result.m_matrix.begin(), result.m_matrix.end(), 0.0f);

        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 4; ++col) {
                result.m_matrix[row * 4 + col] = m_matrix[row * 4] * other.m_matrix[col]
                                                 + m_matrix[row * 4 + 1] * other.m_matrix[4 + col]
                                                 + m_matrix[row * 4 + 2] * other.m_matrix[8 + col];
            }
            result.m_matrix[row * 4 + 3] += m_matrix[row * 4 + 3];
        }
        return result;
    }
};

std::ostream &operator<<(std::ostream &os, Transform const &t)
{
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 4; ++col) { os << t.m_matrix[row * 4 + col] << ", "; }
        os << "\n";
    }
    return os;
}

class Instance
{
  public:
    Instance(OptixTraversableHandle inner, int id, TransformFloats const &trafos)
    {
        m_instance.traversableHandle = inner;
        m_instance.visibilityMask    = 255;
        m_instance.instanceId        = id;
        m_instance.sbtOffset         = 0;
        m_instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
        memcpy(m_instance.transform, trafos.data(), sizeof(float) * 12);

        cudaMalloc(&m_dev_instance, sizeof(OptixInstance));
        update_device();
    }

    void update_device() { cudaMemcpy(m_dev_instance, &m_instance, sizeof(OptixInstance), cudaMemcpyHostToDevice); }

    friend class InstanceGAS;

  private:
    OptixInstance m_instance;
    void         *m_dev_instance;
};


class InstanceGAS
{

  private:
    InstanceGAS() {};

  public:
    static InstanceGAS
        from_trafos(Device const &device, OptixTraversableHandle innerGAS, std::span<TransformFloats> trafos)
    {
        InstanceGAS gas;

        int id = 0;
        for (const auto &trafo : trafos) { gas.m_instances.emplace_back(innerGAS, id++, trafo); }

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags             = OPTIX_BUILD_FLAG_NONE;
        accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

        OptixBuildInput input;

        input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;

        OptixBuildInputInstanceArray *buildInput = &input.instanceArray;

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&buildInput->instances), sizeof(OptixInstance) * gas.m_instances.size()));

        std::vector<OptixInstance> temp_instances;
        for (auto &instance : gas.m_instances) {
            // instance.update_device();
            // temp_instances.push_back(instance.m_dev_instance);
            temp_instances.push_back(instance.m_instance);
        }

        /*
        cudaMemcpy((void *)(buildInput->instances),
            temp_instances.data(),
            sizeof(void *) * gas.m_instances.size(),
            cudaMemcpyHostToDevice);
            */
        CUDA_CHECK(cudaMemcpy((void *)(buildInput->instances),
            temp_instances.data(),
            sizeof(OptixInstance) * gas.m_instances.size(),
            cudaMemcpyHostToDevice));

        buildInput->numInstances   = gas.m_instances.size();
        buildInput->instanceStride = 0;

        OptixAccelBufferSizes gas_buffer_sizes;

        OPTIX_CHECK(optixAccelComputeMemoryUsage(device.get_context(),
            &accel_options,
            &input,
            1,// Number of build inputs
            &gas_buffer_sizes));

        CUdeviceptr d_temp_buffer_gas;
        spdlog::info("preparing buffers");
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&gas.m_d_gas_output_buffer), gas_buffer_sizes.outputSizeInBytes));

        spdlog::info("creating accel handle");
        OPTIX_CHECK(optixAccelBuild(device.get_context(),
            0,// CUDA stream
            &accel_options,
            &input,
            1,// num build inputs
            d_temp_buffer_gas,
            gas_buffer_sizes.tempSizeInBytes,
            gas.m_d_gas_output_buffer,
            gas_buffer_sizes.outputSizeInBytes,
            &gas.m_gas_handle,
            nullptr,// emitted property list
            0// num emitted properties
            ));

        // We can now free the scratch space buffer used during build and the vertex
        // inputs, since they are not needed by our trivial shading method
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer_gas)));
        // CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_vertices)));

        return gas;
    }

    OptixTraversableHandle get_handle() const { return m_gas_handle; }

    size_t instances_count() const { return m_instances.size(); }

  private:
    CUdeviceptr            m_d_gas_output_buffer;
    OptixTraversableHandle m_gas_handle;
    std::vector<Instance>  m_instances;
};
