#ifndef __DIRECTIONAL_LIGHT_H__
#define __DIRECTIONAL_LIGHT_H__

#include "cuda_runtime_api.h"
#include "device_object.h"
#include <cuda/random.h>
#include <sutil/vec_math.h>
#include <vector_types.h>

class DirectionalLight : public DeviceObject<DirectionalLight>
{
  public:
    __host__ DirectionalLight(float3 const &direction, float3 const &lumi, float jitter)
        : m_direction(direction), m_lumi(lumi), m_jitter(jitter)
    {}
    __host__ void set_direction(float3 const &direction) { m_direction = direction; }
    __host__ void set_lumi(float3 const &lumi) { m_lumi = lumi; }

    __host__ __device__ [[nodiscard]] float3 wi(float3 const &p, unsigned int &seed) const
    {
        return m_direction + make_float3(m_jitter * rnd(seed), m_jitter * rnd(seed), m_jitter * rnd(seed));
    }

    __host__ __device__ [[nodiscard]] float3 lumi() const { return m_lumi; }

  private:
    float3 m_direction = { 1.0f, 1.0f, 0.0f };
    float3 m_lumi      = { 1.0f, 1.0f, 1.0f };
    float  m_jitter    = 0.0f;
    float3 m_dark      = { 0.0f, 0.0f, 0.0f };
};

#endif
