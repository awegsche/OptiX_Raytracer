#include "triangle_gas.h"
#include "sutil/Exception.h"
#include "sutil/vec_math.h"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <filesystem>
#include <nbt.h>
#include <stdexcept>
#include <tuple>

#include <optix_function_table_definition.h>
#include <optix_stubs.h>
// #include <optix_function_table_definition.h>
std::tuple<std::vector<float3>, std::vector<float3>, std::vector<int>> load_nbt(const std::string &filename)
{
    std::vector<float3> vertices;
    std::vector<float3> normals;
    std::vector<int>    mat_indices;

    if (!std::filesystem::exists(filename)) { throw std::runtime_error("can't find file"); }

    const auto root = nbt::read_from_file(filename);

    for (const auto &mesh : root.get<nbt::TAG_Compound>()) {
        const nbt::compound &mesh_compound = mesh.get<nbt::TAG_Compound>();

        const std::vector<uint8_t> &vertex_data = mesh_compound["vertices"]->get<nbt::TAG_Byte_Array>();
        const std::vector<uint8_t> &normal_data = mesh_compound["normals"]->get<nbt::TAG_Byte_Array>();

        size_t float3_size = (sizeof(float) * 3);
        size_t nvertices   = vertex_data.size() / float3_size;

        const auto float3_from_bytes = [](std::vector<uint8_t> const &bytes, std::vector<float3> &float3s, size_t i) {
            static size_t float3_size = (sizeof(float) * 3);
            uint32_t      x           = 0;
            float3        vec;

            x |= bytes[float3_size * i];
            x |= bytes[float3_size * i + 1] << 8;
            x |= bytes[float3_size * i + 2] << 16;
            x |= bytes[float3_size * i + 3] << 24;

            vec.x = *reinterpret_cast<float *>(&x);

            x = 0;
            x |= bytes[float3_size * i + 4];
            x |= bytes[float3_size * i + 4 + 1] << 8;
            x |= bytes[float3_size * i + 4 + 2] << 16;
            x |= bytes[float3_size * i + 4 + 3] << 24;

            vec.y = *reinterpret_cast<float *>(&x);

            x = 0;
            x |= bytes[float3_size * i + 8];
            x |= bytes[float3_size * i + 8 + 1] << 8;
            x |= bytes[float3_size * i + 8 + 2] << 16;
            x |= bytes[float3_size * i + 8 + 3] << 24;

            vec.z = *reinterpret_cast<float *>(&x);
            float3s.push_back(vec);
        };

        vertices.reserve(vertices.size() + nvertices);
        normals.reserve(vertices.size() + nvertices);
        for (size_t i = 0; i < nvertices; ++i) {
            float3_from_bytes(vertex_data, vertices, i);
            float3_from_bytes(normal_data, normals, i);
            mat_indices.push_back(0);
        }
    }

    return { vertices, normals, mat_indices };
}

std::tuple<std::vector<float3>, std::vector<float3>, std::vector<int>> load_assimp(const std::string &filename)
{
    std::vector<float3> vertices;
    std::vector<float3> normals;
    std::vector<int>    mat_indices;
    Assimp::Importer    importer;

    spdlog::info("loading bunny");
    const aiScene *scene = importer.ReadFile(filename, 0
        //,aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_GenNormals);
        // aiProcess_CalcTangentSpace | aiProcess_Triangulate | aiProcess_JoinIdenticalVertices |
        // aiProcess_SortByPType);
    );

    spdlog::info("loading finished");

    if (scene == nullptr) {
        spdlog::error("{}", importer.GetErrorString());
        return { vertices, normals, mat_indices };
    }

    spdlog::info("scene has {} meshes", scene->mNumMeshes);

    auto aiVect_to_float3 = [](aiVector3D const &vec, float &floor) -> float3 {
        if (vec.y < floor) { floor = vec.y; }
        return { vec.x, vec.y, vec.z };
    };

    float floor = 1.0e12f;
    float f_    = 0.0f;

    for (size_t i = 0; i < scene->mNumMeshes; ++i) {
        const aiMesh *mesh = scene->mMeshes[i];
        spdlog::info("loading mesh {} / {}", i, scene->mNumMeshes);
        spdlog::info("{} triangles", mesh->mNumFaces);

        for (size_t iface = 0; iface < mesh->mNumFaces; ++iface) {
            const aiFace &face = mesh->mFaces[iface];
            vertices.push_back(aiVect_to_float3(mesh->mVertices[face.mIndices[0]], floor));
            vertices.push_back(aiVect_to_float3(mesh->mVertices[face.mIndices[1]], floor));
            vertices.push_back(aiVect_to_float3(mesh->mVertices[face.mIndices[2]], floor));
            mat_indices.push_back(0);


            if (mesh->HasNormals()) {
                normals.push_back(aiVect_to_float3(mesh->mNormals[face.mIndices[0]], f_));
                normals.push_back(aiVect_to_float3(mesh->mNormals[face.mIndices[1]], f_));
                normals.push_back(aiVect_to_float3(mesh->mNormals[face.mIndices[2]], f_));
            }
        }
    }

    vertices.reserve(vertices.size() * 4);
    // duplicate mesh triangles (TODO: replace this with instancing)
    const int MROWS     = 0;
    const int MCOLS     = 0;
    size_t    nvertices = vertices.size();
    for (int x = 0; x < MROWS; ++x)
        for (int y = 0; y < MCOLS; ++y) {
            if (x == MCOLS / 2 && y == MCOLS / 2) continue;
            for (size_t v = 0; v < nvertices; ++v) {
                float3 new_vert = vertices[v];
                new_vert.x += 0.25f * (x - MCOLS / 2);
                new_vert.z += 0.25f * (y - MROWS / 2);
                vertices.push_back(new_vert);
            }
            for (size_t v = 0; v < nvertices / 3; ++v) { mat_indices.push_back((y)*MROWS + x); }
        }

    // add tesselated floor

    for (int i = -10; i < 10; ++i) {
        for (int j = -10; j < 10; ++j) {
            vertices.push_back({ (float)i * 0.1f, floor, (float)j * 0.1f });
            vertices.push_back({ (float)i * 0.1f, floor, (float)(j + 1) * 0.1f });
            vertices.push_back({ (float)(i + 1) * 0.1f, floor, (float)j * 0.1f });

            vertices.push_back({ (float)(i + 1) * 0.1f, floor, (float)j * 0.1f });
            vertices.push_back({ (float)i * 0.1f, floor, (float)(j + 1) * 0.1f });
            vertices.push_back({ (float)(i + 1) * 0.1f, floor, (float)(j + 1) * 0.1f });

            mat_indices.push_back(26);
            mat_indices.push_back(26);

            for (int n = 0; n < 6; ++n) { normals.push_back({ 0.0, 1.0, 0.0 }); }
        }
    }

    spdlog::info("size of vertices: {:.3} MB", static_cast<float>(vertices.size()) * sizeof(float3) / 1024 / 1024);

    return { vertices, normals, mat_indices };
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
        accel_options.buildFlags             = OPTIX_BUILD_FLAG_NONE;
        accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

        // Triangle build input: simple list of three vertices
        // const auto [vertices, normals, mat_indices] = load_assimp(filename);
        const auto [vertices, normals, mat_indices] = load_nbt(filename);
        m_vertices                                  = vertices;
        m_normals                                   = normals;
        m_mat_indices                               = mat_indices;

        if (m_vertices.size() == 0) { throw std::runtime_error("couldn't load model"); }
        // const std::vector<float3> vertices = make_geometry();
        /*
        const std::array<float3, 3> vertices = {
            { { -0.5f, -0.5f, 0.0f }, { 0.5f, -0.5f, 0.0f }, { 0.0f, 0.5f, 0.0f } }
        };
        */

        const size_t vertices_size = sizeof(float3) * m_vertices.size();
        CUdeviceptr  d_vertices    = 0;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_vertices), vertices_size));
        CUDA_CHECK(
            cudaMemcpy(reinterpret_cast<void *>(d_vertices), m_vertices.data(), vertices_size, cudaMemcpyHostToDevice));

        // Our build input is a simple list of non-indexed triangle vertices
        const uint32_t  triangle_input_flags[1]    = { OPTIX_GEOMETRY_FLAG_NONE };
        OptixBuildInput triangle_input             = {};
        triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input.triangleArray.numVertices   = static_cast<uint32_t>(m_vertices.size());
        triangle_input.triangleArray.vertexBuffers = &d_vertices;
        triangle_input.triangleArray.flags         = triangle_input_flags;
        triangle_input.triangleArray.numSbtRecords = 1;

        OPTIX_CHECK(optixAccelComputeMemoryUsage(device.get_context(),
            &accel_options,
            &triangle_input,
            1,// Number of build inputs
            &m_gas_buffer_sizes));
        CUdeviceptr d_temp_buffer_gas;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp_buffer_gas), m_gas_buffer_sizes.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_d_gas_output_buffer), m_gas_buffer_sizes.outputSizeInBytes));

        OPTIX_CHECK(optixAccelBuild(device.get_context(),
            0,// CUDA stream
            &accel_options,
            &triangle_input,
            1,// num build inputs
            d_temp_buffer_gas,
            m_gas_buffer_sizes.tempSizeInBytes,
            m_d_gas_output_buffer,
            m_gas_buffer_sizes.outputSizeInBytes,
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
