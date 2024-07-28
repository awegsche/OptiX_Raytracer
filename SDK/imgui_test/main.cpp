#include <fmt/core.h>
#include <iostream>
#include <string>

#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>


#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>

#include "device.h"
#include "make_geometry.h"
#include "optixTriangle.h"
#include "tracer_window.h"
#include "triangle_gas.h"


template<typename T> struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using RayGenSbtRecord = SbtRecord<RayGenData>;
using MissSbtRecord = SbtRecord<MissData>;
using HitGroupSbtRecord = SbtRecord<HitGroupData>;

int main(int argc, char *argv[])
{
    CLI::App app("Raytracer");
    std::string outfile;
    std::string modelfile;
    bool gui = false;

    CLI::Option *out_opt = app.add_option("-o,--outfile,outfile", outfile, "render outputfile");
    CLI::Option *model_opt = app.add_option("-m,--model,model", modelfile, "3D model to load");
    CLI::Option *gui_opt = app.add_flag("-w,--window", gui, "Run in windowed mode");

    CLI11_PARSE(app, argc, argv);

    if (modelfile.empty()) {
        spdlog::error("no model file given, nothing to render. ABORT");
        return 1;
    }
    if (outfile.empty()) {
        outfile = "last_render.png";
        spdlog::warn("no outfile given, render to \"{}\"", outfile);
    }

    try {
        Device device;

        spdlog::warn("model file: {}", modelfile);
        TriangleGAS triangles(device, modelfile);
        //
        spdlog::info("Create module");
        //
        OptixModule module = nullptr;
        OptixPipelineCompileOptions pipeline_compile_options = {};
        {
            OptixModuleCompileOptions module_compile_options = {};
#if !defined(NDEBUG)
            module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
            module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

            pipeline_compile_options.usesMotionBlur = false;
            pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
            pipeline_compile_options.numPayloadValues = 3;
            pipeline_compile_options.numAttributeValues = 3;
            pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
            pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
            pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

            size_t inputSize = 0;
            const char *input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixTriangle.cu", inputSize);

            OPTIX_CHECK_LOG(optixModuleCreate(device.get_context(),
                &module_compile_options,
                &pipeline_compile_options,
                input,
                inputSize,
                LOG,
                &LOG_SIZE,
                &module));
        }


        //
        // Create program groups
        //
        OptixProgramGroup raygen_prog_group = nullptr;
        OptixProgramGroup miss_prog_group = nullptr;
        OptixProgramGroup hitgroup_prog_group = nullptr;
        {
            OptixProgramGroupOptions program_group_options = {};// Initialize to zeros

            OptixProgramGroupDesc raygen_prog_group_desc = {};//
            raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_prog_group_desc.raygen.module = module;
            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
            OPTIX_CHECK_LOG(optixProgramGroupCreate(device.get_context(),
                &raygen_prog_group_desc,
                1,// num program groups
                &program_group_options,
                LOG,
                &LOG_SIZE,
                &raygen_prog_group));

            OptixProgramGroupDesc miss_prog_group_desc = {};
            miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            miss_prog_group_desc.miss.module = module;
            miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
            OPTIX_CHECK_LOG(optixProgramGroupCreate(device.get_context(),
                &miss_prog_group_desc,
                1,// num program groups
                &program_group_options,
                LOG,
                &LOG_SIZE,
                &miss_prog_group));

            OptixProgramGroupDesc hitgroup_prog_group_desc = {};
            hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hitgroup_prog_group_desc.hitgroup.moduleCH = module;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
            OPTIX_CHECK_LOG(optixProgramGroupCreate(device.get_context(),
                &hitgroup_prog_group_desc,
                1,// num program groups
                &program_group_options,
                LOG,
                &LOG_SIZE,
                &hitgroup_prog_group));
        }

        //
        // Link pipeline
        //
        OptixPipeline pipeline = nullptr;
        {
            const uint32_t max_trace_depth = 1;
            OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };

            OptixPipelineLinkOptions pipeline_link_options = {};
            pipeline_link_options.maxTraceDepth = max_trace_depth;
            OPTIX_CHECK_LOG(optixPipelineCreate(device.get_context(),
                &pipeline_compile_options,
                &pipeline_link_options,
                program_groups,
                sizeof(program_groups) / sizeof(program_groups[0]),
                LOG,
                &LOG_SIZE,
                &pipeline));

            OptixStackSizes stack_sizes = {};
            for (auto &prog_group : program_groups) {
                OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, pipeline));
            }

            uint32_t direct_callable_stack_size_from_traversal;
            uint32_t direct_callable_stack_size_from_state;
            uint32_t continuation_stack_size;
            OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes,
                max_trace_depth,
                0,// maxCCDepth
                0,// maxDCDEpth
                &direct_callable_stack_size_from_traversal,
                &direct_callable_stack_size_from_state,
                &continuation_stack_size));
            OPTIX_CHECK(optixPipelineSetStackSize(pipeline,
                direct_callable_stack_size_from_traversal,
                direct_callable_stack_size_from_state,
                continuation_stack_size,
                1// maxTraversableDepth
                ));
        }

        //
        spdlog::info("Set up shader binding table");
        //
        OptixShaderBindingTable sbt = {};
        {
            CUdeviceptr raygen_record;
            const size_t raygen_record_size = sizeof(RayGenSbtRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&raygen_record), raygen_record_size));
            RayGenSbtRecord rg_sbt;
            OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void *>(raygen_record), &rg_sbt, raygen_record_size, cudaMemcpyHostToDevice));

            CUdeviceptr miss_record;
            size_t miss_record_size = sizeof(MissSbtRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&miss_record), miss_record_size));
            MissSbtRecord ms_sbt;
            ms_sbt.data = { 0.3f, 0.1f, 0.2f };
            OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
            CUDA_CHECK(
                cudaMemcpy(reinterpret_cast<void *>(miss_record), &ms_sbt, miss_record_size, cudaMemcpyHostToDevice));

            CUdeviceptr hitgroup_record;
            size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&hitgroup_record), hitgroup_record_size));
            HitGroupSbtRecord hg_sbt;
            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void *>(hitgroup_record), &hg_sbt, hitgroup_record_size, cudaMemcpyHostToDevice));

            sbt.raygenRecord = raygen_record;
            sbt.missRecordBase = miss_record;
            sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
            sbt.missRecordCount = 1;
            sbt.hitgroupRecordBase = hitgroup_record;
            sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
            sbt.hitgroupRecordCount = 1;
        }


        CUstream stream;
        CUDA_CHECK(cudaStreamCreate(&stream));


        // ---- setup scene -----------------------------------------------------------------------

        Camera cam{};
        cam.set_eye({ 0.0, 1.0, -10.0 });
        cam.set_up({ 0.0f, 0.0000073f, 1.0f });
        cam.set_lookat({ 0.0, 0.1, 0.0 });
        cam.set_aperture(0.0);
        cam.set_fd(1.0);
        cam.set_fov(45.0);
        cam.compute_uvw();

        PointLight light{};
        light.set_position({ 0.0, 3000.0, 4000.0 });

        Params params;
        params.camera = cam.new_device_ptr();
        params.light = light.new_device_ptr();
        params.vertices = triangles.get_device_vertices();
        params.handle = triangles.get_gas_handle();
        // params.normals = triangles.get_device_normals();
        params.tfactor = 0.5f;
        params.dt = 0;

        /*
        sutil::CUDAOutputBuffer<uchar4> output_buffer(sutil::CUDAOutputBufferType::CUDA_DEVICE, buf_width, buf_height);
        CUDA_CHECK(cudaMalloc(&params.film, sizeof(float3) * buf_width * buf_height));
        */

        if (gui) {
            TracerWindow window{ stream, pipeline, sbt, params };
            window.set_outfile(outfile);
            window.run();
        }

        //
        // Cleanup
        //
        {
            CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.raygenRecord)));
            CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.missRecordBase)));
            CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.hitgroupRecordBase)));
            params.cleanup();

            OPTIX_CHECK(optixPipelineDestroy(pipeline));
            OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_group));
            OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group));
            OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
            OPTIX_CHECK(optixModuleDestroy(module));

            OPTIX_CHECK(optixDeviceContextDestroy(device.get_context()));
        }
    } catch (std::exception &e) {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
