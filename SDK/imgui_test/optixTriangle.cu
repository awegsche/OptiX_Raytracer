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
    params.camera->compute_ray(idx, dim, ray_origin, ray_direction, params.dt);

    // Trace the ray against our scene hierarchy
    unsigned int p0, p1, p2;
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
    float3 result;
    result.x = __uint_as_float(p0);
    result.y = __uint_as_float(p1);
    result.z = __uint_as_float(p2);

    // Record results in our output raster
    const unsigned int index = idx.y * params.image_width + idx.x;
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
    /*
    OptixTraversableHandle gas = optixGetGASTraversableHandle();
    unsigned int primIdx = optixGetPrimitiveIndex();
    unsigned int sbtIdx = optixGetSbtGASIndex();
    float time = optixGetRayTime();


    float3 data[3];
    optixGetTriangleVertexData( gas, primIdx, sbtIdx, time, data );
    */

    unsigned int vertoffset = optixGetPrimitiveIndex() * 3;

    /*
    const float2 uv = optixGetTriangleBarycentrics();
    float3 v0_interp = uv.x * params.normals[vertoffset] + (1.0f - uv.x) * params.normals[vertoffset+1];
    float3 v1_interp = uv.y * params.normals[vertoffset] + (1.0f - uv.y) * params.normals[vertoffset+1];
    */

    float3 v0 = params.vertices[vertoffset + 1] - params.vertices[vertoffset];
    float3 v1 = params.vertices[vertoffset + 2] - params.vertices[vertoffset];

    float3 normal = normalize(cross(v0, v1));

    setPayload(normal);
}
