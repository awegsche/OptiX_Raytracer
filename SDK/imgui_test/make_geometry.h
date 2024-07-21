#include <vector>
#include <vector_types.h>

std::vector<float3> make_geometry() {
    std::vector<float3> geo;

    geo.push_back({0.0, 0.0, -10.0});
    geo.push_back({0.0, 1.0, -10.0});
    geo.push_back({1.0, 0.0, -10.0});

    geo.push_back({1.0, 1.0, -10.0});
    geo.push_back({0.0, 1.0, -10.0});
    geo.push_back({1.0, 0.0, -10.0});

    geo.push_back({0.0, 0.0, -11.0});
    geo.push_back({0.0, 1.0, -11.0});
    geo.push_back({1.0, 0.0, -11.0});

    geo.push_back({1.0, 1.0, -11.0});
    geo.push_back({0.0, 1.0, -11.0});
    geo.push_back({1.0, 0.0, -11.0});

    return geo;
}
