#ifndef __CAMERA_H__
#define __CAMERA_H__

#include "cuda_runtime_api.h"
#include <cuda.h>
#include <cuda/random.h>
#include <cuda_runtime.h>
#include <sutil/vec_math.h>
#include <vector_types.h>

#ifndef __CUDACC__
#include <sutil/Exception.h>
#endif

class Camera
{
  public:
    __host__ void set_eye(float3 new_eye) { eye = new_eye; }

    /** @brief Sets focal distance
     *
     * Note: internal focal distance is relative to `length(lookat - eye)`
     * */
    __host__ void set_fd(float new_fd) { fd = new_fd / length(lookat - eye); }

    __host__ void set_fov(float new_fov) { fov = new_fov; }

    __host__ void set_aperture(float new_aperture) { aperture = new_aperture; }

    __host__ void set_ortho(bool new_ortho) { ortho = new_ortho; }

#ifndef __CUDACC__
    /** @brief Creates a new Camera on the device and returns its dev_ptr.
     * */
    __host__ Camera *new_device_ptr() const
    {
        Camera *ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&ptr, sizeof(Camera)));
        CUDA_CHECK(cudaMemcpy(ptr, this, sizeof(Camera), cudaMemcpyHostToDevice));

        return ptr;
    }
#endif

    __host__ void update_device_ptr(Camera *ptr) const
    {
        cudaMemcpy(ptr, this, sizeof(Camera), cudaMemcpyHostToDevice);
    }

    __host__ void compute_uvw()
    {

        w = lookat - eye;
        w *= fd;
        const float wlen = length(w);

        u = normalize(cross(w, up));
        v = normalize(cross(u, w));

        const float vlen = wlen * tanf(0.5f * fov * M_PIf / 180.0f);
        v *= vlen;
        const float ulen = vlen;
        u *= ulen;
    }

    __host__ __device__ __forceinline__ void
        compute_ray(uint3 idx, uint3 dim, float3 &origin, float3 &direction, unsigned int dt) const
    {
        float2 d = 2.0f
                       * make_float2(static_cast<float>(idx.x) / static_cast<float>(dim.x),
                           static_cast<float>(idx.y) / static_cast<float>(dim.y))
                   - 1.0f;

        unsigned int seed = tea<4>(idx.x + dim.x * idx.y, dt);

        const float2 dx = make_float2((rnd(seed) - 0.5f) * aperture, (rnd(seed) - 0.5f) * aperture);

        d = d - dx;
        direction = normalize(d.x * u + d.y * v + w);
        origin = ortho ? eye + d.x * u + d.y * v : eye;
        origin = origin + dx.x * u + dx.y * v;
    }


  private:
    float3 eye = { 0.0, 1.0, -2.0 };
    float3 u = {};
    float3 v = {};
    float3 w = {};
    bool ortho = false;
    float fov = 45.0f;
    float fd = 1.0f;
    float aperture = 0.0f;

    float3 up = { 0.0, 1.0, 0.000073 };// avoid straight up for singularities
    float3 lookat = { 0.0, 0.0, 0.0 };
};

#endif
