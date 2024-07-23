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

std::vector<float3> load_assimp(const std::string& filename)
{
    std::vector<float3> vertices;
    Assimp::Importer importer;

    spdlog::info("loading bunny");
    const aiScene *scene = importer.ReadFile(filename,
        aiProcess_CalcTangentSpace | aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);

    spdlog::info("loading finished");

    if (scene == nullptr) {
        spdlog::error("{}", importer.GetErrorString());
        return vertices;
    }

    spdlog::info("scene has {} meshes", scene->mNumMeshes);

    auto aiVect_to_float3 = [](aiVector3D const &vec, float& floor) -> float3 {
        if (vec.y < floor) floor = vec.y;
        return { vec.x*5.0f, vec.y*5.0f, vec.z*5.0f }; };

    float floor = 1.0e12f;

    for (size_t i = 0; i < scene->mNumMeshes; ++i) { const aiMesh *mesh = scene->mMeshes[i];
        spdlog::info("loading mesh {} / {}", i, scene->mNumMeshes);
        spdlog::info("{} triangles", mesh->mNumFaces);
        for (size_t iface = 0; iface < mesh->mNumFaces; ++iface) { 
            const aiFace& face = mesh->mFaces[iface];
            vertices.push_back(aiVect_to_float3(mesh->mVertices[face.mIndices[0]], floor)); 
            vertices.push_back(aiVect_to_float3(mesh->mVertices[face.mIndices[1]], floor)); 
            vertices.push_back(aiVect_to_float3(mesh->mVertices[face.mIndices[2]], floor)); 
        }
    }


    // add tesselated floor

    for (int i = -10; i < 10; ++i) {
        for (int j = -10; j < 10; ++j) {
            vertices.push_back({ (float)i * 0.1f, floor, (float)j * 0.1f });
            vertices.push_back({ (float)(i+1) * 0.1f, floor, (float)j * 0.1f });
            vertices.push_back({ (float)i * 0.1f, floor, (float)(j+1) * 0.1f });

            vertices.push_back({ (float)(i+1) * 0.1f, floor, (float)j * 0.1f });
            vertices.push_back({ (float)i * 0.1f, floor, (float)(j+1) * 0.1f });
            vertices.push_back({ (float)(i+1) * 0.1f, floor, (float)(j+1) * 0.1f });
        }
    }



    return vertices;
}

