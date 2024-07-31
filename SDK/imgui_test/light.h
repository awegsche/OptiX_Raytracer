#ifndef __LIGHT_H__
#define __LIGHT_H__

#include "cuda_runtime_api.h"
#include "directional_light.h"
#include "point_light.h"
#include "volumetric_light.h"
#include <vector_types.h>

class LightVariant
{
  public:
    __host__ LightVariant(VolumetricLight const &v) : _tag(Tag::Volumetric) { _payload.v = v; }

    __host__ LightVariant(PointLight const &p) : _tag(Tag::Point) { _payload.p = p; }

    __host__ LightVariant(DirectionalLight const &d) : _tag(Tag::Directional) { _payload.d = d; }

    __host__ __device__ [[nodiscard]] float3 wi(float3 const &p, unsigned int &seed) const
    {
        switch (_tag) {
        case Tag::Point:
            return _payload.p.wi(p, seed);
        case Tag::Directional:
            return _payload.d.wi(p, seed);
        case Tag::Volumetric:
            return _payload.v.wi(p, seed);
        }
    }
    __host__ __device__ [[nodiscard]] float3 lumi() const
    {
        switch (_tag) {
        case Tag::Point:
            return _payload.p.lumi();
        case Tag::Directional:
            return _payload.d.lumi();
        case Tag::Volumetric:
            return _payload.v.lumi();
        }
    }

  private:
    union Payload {
        PointLight       p;
        DirectionalLight d;
        VolumetricLight  v;

        Payload() : p() {}
    } _payload;
    enum class Tag { Point, Directional, Volumetric } _tag{};
};

#endif
