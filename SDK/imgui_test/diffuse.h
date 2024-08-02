#include <vector_types.h>

class DiffuseMaterial
{
  public:
    explicit DiffuseMaterial(float3 color) : m_color(color) {}

    __host__ __device__ [[nodiscard]] float3
             f(const float3 &p, const float3 &wi, const float3 &wo, unsigned int &seed) const
    {
        return m_color;
    }

  private:
    float3 m_color = { 0.5f, 0.5f, 0.5f };
};
