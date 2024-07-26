//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

// #include <__clang_cuda_runtime_wrapper.h>
#include <optix.h>

#include "camera.h"
#include "internal/optix_micromap_impl.h"
#include "optixTriangle.h"
#include "optix_device.h"
#include <cuda/helpers.h>
#include <cuda/random.h>

#include <sutil/vec_math.h>

extern "C" {
__constant__ Params params;
}

static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, float3 &p)
{
    // Uniformly sample disk.
    const float r = sqrtf(u1);
    const float phi = 2.0f * M_PIf * u2;
    p.x = r * cosf(phi);
    p.y = r * sinf(phi);

    // Project up to hemisphere.
    p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
}

struct Onb
{
    __forceinline__ __device__ Onb(const float3 &normal)
    {
        m_normal = normal;

        if (fabs(m_normal.x) > fabs(m_normal.z)) {
            m_binormal.x = -m_normal.y;
            m_binormal.y = m_normal.x;
            m_binormal.z = 0;
        } else {
            m_binormal.x = 0;
            m_binormal.y = -m_normal.z;
            m_binormal.z = m_normal.y;
        }

        m_binormal = normalize(m_binormal);
        m_tangent = cross(m_binormal, m_normal);
    }

    __forceinline__ __device__ void inverse_transform(float3 &p) const
    {
        p = p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
    }

    float3 m_tangent;
    float3 m_binormal;
    float3 m_normal;
};

static __forceinline__ __device__ void setPayload(float3 p)
{
    optixSetPayload_0(__float_as_uint(p.x));
    optixSetPayload_1(__float_as_uint(p.y));
    optixSetPayload_2(__float_as_uint(p.z));
}

extern "C" __global__ void __raygen__rg()
{
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();


    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen
    float3 ray_origin, ray_direction;
    // computeRay(params.camera, idx, dim, ray_origin, ray_direction);

    // Trace the ray against our scene hierarchy
    unsigned int p0, p1, p2;
    const unsigned int index = idx.y * params.image_width + idx.x;
    float3 result = { 0.0, 0.0, 0.0 };

    unsigned int seed = tea<4>(idx.x + dim.x * idx.y, params.dt);

    for (unsigned int i = 0; i < params.samples_per_frame; ++i) {
        params.camera->compute_ray(idx, dim, ray_origin, ray_direction, seed);
        optixTrace(params.handle,
            ray_origin,
            ray_direction,
            0.0f,// Min intersection distance
            1e16f,// Max intersection distance
            0.0f,// rayTime -- used for motion blur
            OptixVisibilityMask(255),// Specify always visible
            OPTIX_RAY_FLAG_NONE,
            0,// SBT offset   -- See SBT discussion
            1,// SBT stride   -- See SBT discussion
            0,// missSBTIndex -- See SBT discussion
            p0,
            p1,
            p2);
        result.x += __uint_as_float(p0);
        result.y += __uint_as_float(p1);
        result.z += __uint_as_float(p2);
    }

    // Record results in our output raster
    if (params.dirty) {
        params.film[index] = result;
    } else {
        params.film[index] = params.film[index] + result;
    }
    params.image[index] = make_color(params.film[index] / static_cast<float>(params.dt));
}


extern "C" __global__ void __miss__ms()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    // MissData* miss_data  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    float3 result;
    result.x = static_cast<float>(idx.x) / static_cast<float>(dim.x);
    result.y = static_cast<float>(idx.y) / static_cast<float>(dim.y);
    result.z = static_cast<float>(idx.z) / static_cast<float>(dim.z);
    /*
    unsigned int seed = tea<4>(idx.x + idx.y*dim.x, params.dt);
    const float rand = rnd(seed);
    const float3 result = make_float3(rand, rand, rand);
    */
    setPayload(result);
}


extern "C" __global__ void __closesthit__ch()
{
    // When built-in triangle intersection is used, a number of fundamental
    // attributes are provided by the OptiX API, indlucing barycentric coordinates.

    // TODO: lookup diffuse shading in PBRT / RTFTGU and implement it here.

    // calc normal
    unsigned int vertoffset = optixGetPrimitiveIndex() * 3;

    float3 v0 = params.vertices[vertoffset + 1] - params.vertices[vertoffset];
    float3 v1 = params.vertices[vertoffset + 2] - params.vertices[vertoffset];

    float3 normal = normalize(cross(v0, v1));

    const float3 P = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection() + normal * 0.0001f;

    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    unsigned int seed = tea<4>(idx.x + dim.x * idx.y, params.dt);

    // jittered light dir
    float3 light_dir = { 0.71f + rnd(seed) * 0.01f, 0.51f + rnd(seed) * 0.01f, 0.0f + rnd(seed) * 0.01f };
    light_dir = normalize(light_dir);

    optixTraverse(params.handle,
        P,
        light_dir,
        0.01,
        1e16f,
        0.0f,// rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0,// SBT offset
        1,// SBT stride
        0// missSBTIndex
    );

    const float3 shadow_color = { 0.0f, 0.0f, 0.0f };
    const float3 light_color = { 0.25f, 0.25f, 0.25f };

    float3 result = optixHitObjectIsHit() ? shadow_color : light_color;

    Onb onb(normal);

    const float u1 = rnd(seed);
    const float u2 = rnd(seed);
    float3 out;
    cosine_sample_hemisphere(u1, u2, out);

    onb.inverse_transform(out);

    // We are only casting probe rays so no shader invocation is needed
    optixTraverse(params.handle,
        P,
        out,
        0.01,
        1e16f,
        0.0f,// rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0,// SBT offset
        1,// SBT stride
        0// missSBTIndex
    );

    result += optixHitObjectIsHit() ? shadow_color : light_color;
    setPayload(result);
}
