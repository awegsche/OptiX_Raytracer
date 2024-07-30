#ifndef __VOLUMETRIC_LIGHT_H__
#define __VOLUMETRIC_LIGHT_H__

#include "cuda_runtime_api.h"
#include "device_object.h"
#include "light.h"
#include <cuda/random.h>
#include <sutil/vec_math.h>
#include <vector_types.h>

class VolumetricLight
    : public DeviceObject<VolumetricLight>
    //, public Light
{
  public:
    __host__ void set_position(float3 const &position) { m_position = position; }
    __host__ void set_lumi(float3 const &lumi) { m_lumi = lumi; }
    __host__ void set_radius(float radius) { m_radius = radius; }

    __host__ __device__ [[nodiscard]] float3 wi(float3 const &p, unsigned int &seed) const 
    {
        return m_position + make_float3(m_radius * rnd(seed), m_radius * rnd(seed), m_radius * rnd(seed)) - p;
    }

    __host__ __device__ [[nodiscard]] float3 lumi() const  { return m_lumi; }

  private:
    float3 m_position = { 1.0f, 1.0f, 0.0f };
    float  m_radius   = 1.0f;
    float3 m_lumi     = { 1.0f, 1.0f, 1.0f };
    float3 m_dark     = { 0.0f, 0.0f, 0.0f };
};

#endif
