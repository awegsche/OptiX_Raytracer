#include <vector>
#include <string>

#include <spdlog/spdlog.h>
#include <vector_types.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

std::vector<float3> make_geometry()
{
    std::vector<float3> geo;

    geo.push_back({ 0.0, 0.0, -10.0 });
    geo.push_back({ 0.0, 1.0, -10.0 });
    geo.push_back({ 1.0, 0.0, -10.0 });

    geo.push_back({ 1.0, 1.0, -10.0 });
    geo.push_back({ 0.0, 1.0, -10.0 });
    geo.push_back({ 1.0, 0.0, -10.0 });

    geo.push_back({ 0.0, 0.0, -11.0 });
    geo.push_back({ 0.0, 1.0, -11.0 });
    geo.push_back({ 1.0, 0.0, -11.0 });

    geo.push_back({ 1.0, 1.0, -11.0 });
    geo.push_back({ 0.0, 1.0, -11.0 });
    geo.push_back({ 1.0, 0.0, -11.0 });

    return geo;
}

