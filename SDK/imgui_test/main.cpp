
#include <algorithm>
#include <cmath>
#include <fmt/core.h>
#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <sampleConfig.h>

#include <stdexcept>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/sutil.h>

#include "CLI/CLI.hpp"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "optixTriangle.h"

#include <array>
#include <iostream>
#include <string>

#include <sutil/Camera.h>
#include <sutil/Trackball.h>

#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#include <spdlog/spdlog.h>
#include <CLI/CLI.hpp>

#include "device.h"
#include "make_geometry.h"
#include "triangle_gas.h"

using sutil::GLDisplay;

#ifdef NDEBUG
constexpr int DOWNSAMPLING = 1;
#else
constexpr int DOWNSAMPLING = 4;
#endif
constexpr int32_t side_panel_width = 256;
constexpr int window_width = 1024;
constexpr int window_height = 768;

template<typename T> struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using RayGenSbtRecord = SbtRecord<RayGenData>;
using MissSbtRecord = SbtRecord<MissData>;
using HitGroupSbtRecord = SbtRecord<HitGroupData>;

bool stop_to_render = false;
bool render_to_file = false;

void initGL()
{
    spdlog::debug("init GL");
    if (gladLoadGL() == 0) throw std::runtime_error("Failed to initialize GL");

    GL_CHECK(glClearColor(0.212f, 0.271f, 0.31f, 1.0f));
    GL_CHECK(glClear(GL_COLOR_BUFFER_BIT));
}


static void errorCallback(int error, const char *description)
{
    std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

static void keyCallback(GLFWwindow *window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/)
{
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE) { glfwSetWindowShouldClose(window, true); }
        if (key == GLFW_KEY_SPACE) { stop_to_render = true; }
    }

    if (action == GLFW_RELEASE) {
        if (key == GLFW_KEY_SPACE) {
            stop_to_render = false;
            render_to_file = true;
        }
    }
}

int main(int argc, char *argv[])
{
    CLI::App app("Raytracer");
    std::string outfile;
    std::string modelfile;

    CLI::Option * out_opt = app.add_option("-o,--outfile,outfile", outfile, "render outputfile");
    CLI::Option * model_opt = app.add_option("-m,--model,model", modelfile, "3D model to load");

    CLI11_PARSE(app, argc, argv);

    if (modelfile.empty()) {
        spdlog::error("no model file given, nothing to render. ABORT");
        return 1;
    }
    if (outfile.empty()) {
        outfile = "last_render.png";
        spdlog::warn("no outfile given, render to \"{}\"", outfile);
    }

    float fov = 45.0f;
    float fod = 2.0f;
    int spf = 8;
    float aperture = 0.0f;
    bool orhto = false;
    int width = window_width - side_panel_width;
    int height = window_height;

    int buf_width = width / DOWNSAMPLING;
    int buf_height = height / DOWNSAMPLING;

    try {

        Device device;

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

        sutil::CUDAOutputBuffer<uchar4> output_buffer(sutil::CUDAOutputBufferType::CUDA_DEVICE, buf_width, buf_height);

        CUstream stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        Camera cam{};
        cam.set_eye({ 0.0, 1.0, -10.0 });
        cam.set_lookat({ 0.0, 0.1, 0.0 });
        cam.set_aperture(0.0);
        cam.set_fd(1.0);
        cam.set_fov(45.0);
        cam.compute_uvw();

        Params params;
        params.camera = cam.new_device_ptr();
        params.vertices = triangles.get_device_vertices();
        // params.normals = triangles.get_device_normals();
        params.tfactor = 0.5f;
        params.dt = 0;

        CUDA_CHECK(cudaMalloc(&params.film, sizeof(float3) * buf_width * buf_height));

#ifdef NDEBUG
        spf = 5;
#else
        spf = 1;
#endif
        params.image = output_buffer.map();
        params.image_width = buf_width;
        params.image_height = buf_height;
        params.handle = triangles.get_gas_handle();

        // gui
        sutil::ImageBuffer buffer;

        buffer.data = output_buffer.getHostPointer();
        buffer.width = buf_width;
        buffer.height = buf_height;
        buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;


        //
        // Initialize GLFW state
        //
        GLFWwindow *window = nullptr;
        glfwSetErrorCallback(errorCallback);
        if (!glfwInit()) throw std::runtime_error("Failed to initialize GLFW");
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);// To make Apple happy -- should not be needed
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        window = glfwCreateWindow(window_width, height, "sfef", nullptr, nullptr);
        if (!window) throw std::runtime_error("Failed to create GLFW window");
        glfwMakeContextCurrent(window);
        glfwSetKeyCallback(window, keyCallback);


        //
        // Initialize GL state
        //
        initGL();
        GLDisplay display(buffer.pixel_format);

        GLuint pbo = 0u;
        GL_CHECK(glGenBuffers(1, &pbo));
        GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, pbo));
        GL_CHECK(glBufferData(GL_ARRAY_BUFFER,
            pixelFormatSize(buffer.pixel_format) * buffer.width * buffer.height,
            buffer.data,
            GL_STREAM_DRAW));
        GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));
        // sutil::displayBufferWindow(argv[0], buffer);
        //
        // Display loop
        //

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::StyleColorsDark();
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 130");

        int framebuf_res_x = 0, framebuf_res_y = 0;
        int step = 0;

        spdlog::info("start loop");
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            // glfwWaitEvents();

            if (render_to_file) {
                render_to_file = false;

                sutil::saveImage(outfile.c_str(), buffer, false);
            }

            ++step;
            if (params.dirty) params.dt = 0;

            params.dirty = false;

            if (!stop_to_render) {
                const float phase = static_cast<float>(step) * 1.0e-2;
                cam.set_eye({ 2.0f * sin(phase), 0.5f, 2.0f * cos(phase) });
                params.dirty = true;
            }

            // render
            cam.set_fov(fov);
            cam.set_fd(fod);
            cam.set_ortho(orhto);
            cam.set_aperture(aperture);
            cam.compute_uvw();
            cam.update_device_ptr(params.camera);

            params.image = output_buffer.map();
            params.image_width = buf_width;
            params.image_height = buf_height;
            params.handle = triangles.get_gas_handle();

            params.frame_step();

            const CUdeviceptr d_param = params.to_device();

            OPTIX_CHECK(
                optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt, buf_width, buf_height, /*depth=*/1));
            CUDA_SYNC_CHECK();

            output_buffer.unmap();
            CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_param)));

            buffer.data = output_buffer.getHostPointer();

            GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, pbo));
            GL_CHECK(glBufferData(GL_ARRAY_BUFFER,
                pixelFormatSize(buffer.pixel_format) * buffer.width * buffer.height,
                buffer.data,
                GL_STREAM_DRAW));
            GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            ImGui::SetNextWindowPos({ 0, 0 }, ImGuiCond_::ImGuiCond_Always);
            ImGui::SetNextWindowSize(
                { static_cast<float>(side_panel_width), static_cast<float>(height) }, ImGuiCond_Always);

#ifdef NDEBUG
            ImGui::Begin(
                "Release", 0, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
#else
            ImGui::Begin("Debug", 0, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
#endif
            {
                if (ImGui::CollapsingHeader("Window")) {
                    ImGui::Text("Window Width: %d", window_width);
                    ImGui::Text("Window Height: %d", height);

                    ImGui::Text("Buffer Width: %d", width);
                    ImGui::Text("Buffer Height: %d", height);

                    ImGui::Text("Step: %d", step);

                    ImGui::Text("Camera:");
                    // ImGui::Text("(%.2f, %.2f, %.2f", params.cam_eye.x, params.cam_eye.y, params.cam_eye.z);
                }
            }

            if (ImGui::CollapsingHeader("Settings")) {
                params.dirty |= ImGui::Checkbox("Orthographic", &orhto);
                params.dirty |= ImGui::SliderFloat("T Factor", &params.tfactor, 0.0, 1.0);
                params.dirty |= ImGui::SliderFloat("FOV", &fov, 1.0f, 180.0f);
                params.dirty |= ImGui::SliderFloat("Aperture", &aperture, 0.0f, 1.0f);
                params.dirty |= ImGui::SliderFloat("Focal length", &fod, 0.2f, 10.0f);
                params.dirty |= ImGui::SliderInt("SPF", &spf, 0, 10);
                params.samples_per_frame = static_cast<unsigned int>(pow(2, spf));
                ImGui::Text("Samples per Frame: %d", params.samples_per_frame);
            }

            device.imgui();
            triangles.imgui();

            ImGui::End();

            ImGui::Render();

            // cam.

            glfwGetFramebufferSize(window, &framebuf_res_x, &framebuf_res_y);
            display.display(buffer.width,
                buffer.height,
                side_panel_width,
                0,
                framebuf_res_x - side_panel_width,
                framebuf_res_y,
                pbo);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(window);
        }

        glfwDestroyWindow(window);
        glfwTerminate();

        //
        // Cleanup
        //
        {
            CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.raygenRecord)));
            CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.missRecordBase)));
            CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.hitgroupRecordBase)));
            Params::cleanup(params);

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
