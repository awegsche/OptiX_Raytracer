#ifndef __LIGHT_H__
#define __LIGHT_H__

#include "cuda_runtime_api.h"
#include <vector_types.h>

/**
 * @class Light
 * @brief Light base class.
 *
 * Defines two functions `wi` returning a vector to the light source (possibly jittered) and
 * `lumi` returning the luminosity.
 *
 */
class Light
{
  public:
    Light()          = default;
    Light(const Light &)            = default;
    Light(Light &&)                 = delete;
    Light &operator=(const Light &) = default;
    Light &operator=(Light &&)      = delete;
    virtual ~Light()                = default;

    __host__ __device__ [[nodiscard]] virtual float3 wi(float3 const &p, unsigned int & /*seed*/) const = 0;

    __host__ __device__ [[nodiscard]] virtual float3 lumi() const = 0;
};

#endif
