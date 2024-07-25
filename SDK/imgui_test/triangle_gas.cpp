#include "triangle_gas.h"
#include "sutil/Exception.h"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <stdexcept>
#include <tuple>

#include <optix_function_table_definition.h>
#include <optix_stubs.h>
// #include <optix_function_table_definition.h>

std::tuple<std::vector<float3>, std::vector<float3>> load_assimp(const std::string &filename)
{
    std::vector<float3> vertices;
    std::vector<float3> normals;
    Assimp::Importer importer;

    spdlog::info("loading bunny");
    const aiScene *scene = importer.ReadFile(filename,
        aiProcess_CalcTangentSpace | aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);

    spdlog::info("loading finished");

    if (scene == nullptr) {
        spdlog::error("{}", importer.GetErrorString());
        return { vertices, normals };
    }

    spdlog::info("scene has {} meshes", scene->mNumMeshes);

    auto aiVect_to_float3 = [](aiVector3D const &vec, float &floor) -> float3 {
        if (vec.y < floor) { floor = vec.y; }
        return { vec.x, vec.y, vec.z };
    };

    float floor = 1.0e12f;
    float f_ = 0.0f;

    for (size_t i = 0; i < scene->mNumMeshes; ++i) {
        const aiMesh *mesh = scene->mMeshes[i];
        spdlog::info("loading mesh {} / {}", i, scene->mNumMeshes);
        spdlog::info("{} triangles", mesh->mNumFaces);

        for (size_t iface = 0; iface < mesh->mNumFaces; ++iface) {
            const aiFace &face = mesh->mFaces[iface];
            vertices.push_back(aiVect_to_float3(mesh->mVertices[face.mIndices[0]], floor));
            vertices.push_back(aiVect_to_float3(mesh->mVertices[face.mIndices[1]], floor));
            vertices.push_back(aiVect_to_float3(mesh->mVertices[face.mIndices[2]], floor));

            if (mesh->HasNormals()) {
                normals.push_back(aiVect_to_float3(mesh->mNormals[face.mIndices[0]], f_));
                normals.push_back(aiVect_to_float3(mesh->mNormals[face.mIndices[1]], f_));
                normals.push_back(aiVect_to_float3(mesh->mNormals[face.mIndices[2]], f_));
            }
        }
    }


    // add tesselated floor

    for (int i = -10; i < 10; ++i) {
        for (int j = -10; j < 10; ++j) {
            vertices.push_back({ (float)i * 0.1f, floor, (float)j * 0.1f });
            vertices.push_back({ (float)(i + 1) * 0.1f, floor, (float)j * 0.1f });
            vertices.push_back({ (float)i * 0.1f, floor, (float)(j + 1) * 0.1f });

            vertices.push_back({ (float)(i + 1) * 0.1f, floor, (float)j * 0.1f });
            vertices.push_back({ (float)i * 0.1f, floor, (float)(j + 1) * 0.1f });
            vertices.push_back({ (float)(i + 1) * 0.1f, floor, (float)(j + 1) * 0.1f });

            for (int n = 0; n < 6; ++n) { normals.push_back({ 0.0, 1.0, 0.0 }); }
        }
    }

    spdlog::info("size of vertices: {:.3} MB", static_cast<float>(vertices.size()) * sizeof(float3) / 1024 / 1024);

    return { vertices, normals };
}

TriangleGAS::TriangleGAS(const Device &device, const std::string &filename)
{
    //
    spdlog::info("accel handling");
    //
    {
        // Use default options for simplicity.  In a real use case we would want to
        // enable compaction, etc
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        // Triangle build input: simple list of three vertices
        const auto [vertices, normals] = load_assimp("C:/Users/andiw/3D Objects/bunny/reconstruction/bun_zipper.ply");
        m_vertices = vertices;
        m_normals = normals;

        if (m_vertices.size() == 0) { throw std::runtime_error("couldn't load model"); }
        // const std::vector<float3> vertices = make_geometry();
        /*
        const std::array<float3, 3> vertices = {
            { { -0.5f, -0.5f, 0.0f }, { 0.5f, -0.5f, 0.0f }, { 0.0f, 0.5f, 0.0f } }
        };
        */

        const size_t vertices_size = sizeof(float3) * m_vertices.size();
        CUdeviceptr d_vertices = 0;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_vertices), vertices_size));
        CUDA_CHECK(
            cudaMemcpy(reinterpret_cast<void *>(d_vertices), m_vertices.data(), vertices_size, cudaMemcpyHostToDevice));

        // Our build input is a simple list of non-indexed triangle vertices
        const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
        OptixBuildInput triangle_input = {};
        triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input.triangleArray.numVertices = static_cast<uint32_t>(m_vertices.size());
        triangle_input.triangleArray.vertexBuffers = &d_vertices;
        triangle_input.triangleArray.flags = triangle_input_flags;
        triangle_input.triangleArray.numSbtRecords = 1;

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(device.get_context(),
            &accel_options,
            &triangle_input,
            1,// Number of build inputs
            &gas_buffer_sizes));
        CUdeviceptr d_temp_buffer_gas;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_d_gas_output_buffer), gas_buffer_sizes.outputSizeInBytes));

        OPTIX_CHECK(optixAccelBuild(device.get_context(),
            0,// CUDA stream
            &accel_options,
            &triangle_input,
            1,// num build inputs
            d_temp_buffer_gas,
            gas_buffer_sizes.tempSizeInBytes,
            m_d_gas_output_buffer,
            gas_buffer_sizes.outputSizeInBytes,
            &m_gas_handle,
            nullptr,// emitted property list
            0// num emitted properties
            ));

        // We can now free the scratch space buffer used during build and the vertex
        // inputs, since they are not needed by our trivial shading method
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer_gas)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_vertices)));
    }
}
TriangleGAS::~TriangleGAS() { CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_d_gas_output_buffer))); }
