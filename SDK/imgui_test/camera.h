#ifndef __CAMERA_H__
#define __CAMERA_H__

#include "cuda_runtime_api.h"
#include <cuda.h>
#include <cuda/random.h>
#include <cuda_runtime.h>
#include <sutil/vec_math.h>
#include <vector_types.h>

/**
 * @class Camera
 * @brief Camera class. Calculates primary rays to shoot into the scene.
 *
 */
class Camera
{
  public:
    __host__ void set_lookat(float3 new_lookat) { m_lookat = new_lookat; }

    __host__ void set_eye(float3 new_eye) { m_eye = new_eye; }

    /** @brief Sets focal distance
     *
     * Note: internal focal distance is relative to `length(lookat - eye)`
     * */
    __host__ void set_fd(float new_fd) { m_fd = new_fd / length(m_lookat - m_eye); }

    /**
     * @brief Sets the field of view (angle of visible field)
     *
     * @param new_fov
     * @return
     */
    __host__ void set_fov(float new_fov) { m_fov = new_fov; }

    /**
     * @brief Sets the aperture
     *
     * @param new_aperture
     * @return
     */
    __host__ void set_aperture(float new_aperture) { m_aperture = new_aperture; }

    /**
     * @brief Sets / Unsets orthographic projection
     *
     * @param new_ortho
     * @return
     */
    __host__ void set_ortho(bool new_ortho) { m_ortho = new_ortho; }

    __host__ void move_forward() {
        m_eye += w * m_speed;
    }
    __host__ void move_backward() {
        m_eye -= w * m_speed;
    }
    __host__ void move_left() {
        m_eye -= u * m_speed;
        m_lookat -= u * m_speed;
    }
    __host__ void move_right() {
        m_eye += u * m_speed;
        m_lookat += u * m_speed;
    }
    __host__ void move_up() {
        m_eye += v * m_speed;
        m_lookat += v * m_speed;
    }
    __host__ void move_down() {
        m_eye -= v * m_speed;
        m_lookat -= v * m_speed;
    }
    __host__ void turn_up() {
        m_lookat += v * m_speed;
    }
    __host__ void turn_down() {
        m_lookat -= v * m_speed;
    }
    __host__ void turn_right() {
        m_lookat += u * m_speed;
    }
    __host__ void turn_left() {
        m_lookat -= u * m_speed;
    }

    /**
     * @brief Allocates Camera on device memory and returns the pointer to it
     *
     * @return
     */
    __host__ Camera *new_device_ptr() const
    {
        Camera *ptr = nullptr;
        cudaMalloc(&ptr, sizeof(Camera));
        cudaMemcpy(ptr, this, sizeof(Camera), cudaMemcpyHostToDevice);

        return ptr;
    }

    /**
     * @brief Updates the device side camera object.
     * Must be called whenever the camera changes behaviour / position.
     *
     * @param ptr The device ptr
     * @return
     */
    __host__ void update_device_ptr(Camera *ptr) const
    {
        cudaMemcpy(ptr, this, sizeof(Camera), cudaMemcpyHostToDevice);
    }

    /**
     * @brief Computes ortho basis (not orthonormal, w carries info about focal distance)
     *
     * @return
     */
    __host__ void compute_uvw()
    {
        w = m_lookat - m_eye;
        w *= m_fd;
        const float wlen = length(w);

        u = normalize(cross(w, m_up));
        v = normalize(cross(u, w));

        const float vlen = wlen * tanf(0.5f * m_fov * M_PIf / 180.0f);
        v *= vlen;
        const float ulen = vlen;
        u *= ulen;
    }

    /**
     * @brief Calculates a primary ray for the given pixel.
     *
     * @param idx
     * @param dim
     * @param origin
     * @param direction
     * @param dt
     */
    __host__ __device__ __forceinline__ void
        compute_ray(uint3 idx, uint3 dim, float3 &origin, float3 &direction, unsigned int &seed) const
    {
        float2 d = 2.0f
                       * make_float2(static_cast<float>(idx.x) / static_cast<float>(dim.x),
                           static_cast<float>(idx.y) / static_cast<float>(dim.y))
                   - 1.0f;

        if (m_ortho) {
            direction = normalize(d.x * u + d.y * v + w);
            origin = m_eye + d.x * u + d.y * v;
        } else {
            const float2 dx = make_float2((rnd(seed) - 0.5f) * m_aperture, (rnd(seed) - 0.5f) * m_aperture);
            d = d - dx;
            direction = normalize(d.x * u + d.y * v + w);
            origin = m_eye + dx.x * u + dx.y * v;
        }
    }

  private:
    // position and orientation
    float3 m_eye = { 0.0, 1.0, -2.0 };
    float3 m_lookat = { 0.0, 0.0, 0.0 };
    float3 m_up = { 0.0, 1.0, 0.000073 };// avoid straight up for singularities
                                         //
    // camera settings
    bool m_ortho = false;
    float m_fov = 45.0f;
    float m_fd = 1.0f;
    float m_aperture = 0.0f;

    // movement
    float m_speed = 0.01f;

    // orthogonal basis
    float3 u = {};
    float3 v = {};
    float3 w = {};
};

#endif
