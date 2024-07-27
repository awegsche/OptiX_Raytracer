#ifndef __PARAMS_H__
#define __PARAMS_H__
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
//
#include "camera.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "optix_types.h"
#include "vector_types.h"

struct Params
{
    // settings
    unsigned int image_width = 400;
    unsigned int image_height = 300;
    unsigned int samples_per_frame = 4;

    Camera *camera = nullptr;

    // state 
    unsigned int dt = 0;
    bool dirty = true; // redraw?

    // buffers
    uchar4 *image = nullptr;
    float3 *film = nullptr;

    float tfactor = 1.0;
    OptixTraversableHandle handle = 0;

    // geometry data
    float3* normals = nullptr;
    float3* vertices = nullptr;

    __host__ static void cleanup(Params& params) {
        cudaFree(params.film);
        cudaFree(params.camera);
        cudaFree(params.normals);
    }

    __host__ CUdeviceptr to_device() const {
        CUdeviceptr ptr;
        cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(Params));
        cudaMemcpy(reinterpret_cast<void*>(ptr), this, sizeof(Params), cudaMemcpyHostToDevice);

        return ptr;
    }

    __host__ void alloc_film() {
        cudaMalloc(&film, sizeof(float3) * image_width * image_height);
    }

    __host__ void frame_step() { dt += samples_per_frame; }
};


struct RayGenData
{
    // No data needed
};


struct MissData
{
    float3 bg_color;
};


struct HitGroupData
{
    // No data needed
};

#endif
