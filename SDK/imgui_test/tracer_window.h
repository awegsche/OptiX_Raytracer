#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#include <spdlog/spdlog.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Trackball.h>

#include <GLFW/glfw3.h>
#include <glad/glad.h>

#include <utility>

#include "camera.h"
#include "device.h"
#include "optixTriangle.h"
#include "optix_types.h"
#include "sutil/CUDAOutputBuffer.h"
#include "sutil/sutil.h"
#include "triangle_gas.h"

using sutil::GLDisplay;

constexpr int32_t SIDE_PANEL_WIDTH = 256;
constexpr int WINDOW_WIDTH = 1024;
constexpr int WINDOW_HEIGHT = 768;

#ifdef NDEBUG
constexpr int DOWNSAMPLING = 1;
#else
constexpr int DOWNSAMPLING = 4;
#endif

bool render_to_file = false;

struct Moving
{
    bool forward = false;
    bool backward = false;
    bool left = false;
    bool right = false;
    bool turn_left = false;
    bool turn_right = false;
    bool turn_up = false;
    bool turn_down = false;
    bool up = false;
    bool down = false;
};

Moving moving{};

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
        switch (key) {
        case GLFW_KEY_Q:
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, true);
            break;
        case GLFW_KEY_SPACE:
            render_to_file = true;
            break;
        case GLFW_KEY_W:
            moving.forward = true;
            break;
        case GLFW_KEY_S:
            moving.backward = true;
            break;
        case GLFW_KEY_A:
            moving.left = true;
            break;
        case GLFW_KEY_D:
            moving.right = true;
            break;
        case GLFW_KEY_LEFT:
            moving.turn_left = true;
            break;
        case GLFW_KEY_RIGHT:
            moving.turn_right = true;
            break;
        case GLFW_KEY_C:
            moving.up = true;
            break;
        case GLFW_KEY_V:
            moving.down = true;
            break;
        case GLFW_KEY_UP:
            moving.turn_up = true;
            break;
        case GLFW_KEY_DOWN:
            moving.turn_down = true;
            break;
        }
    }
    if (action == GLFW_RELEASE) {
        switch (key) {
        case GLFW_KEY_W:
            moving.forward = false;
            break;
        case GLFW_KEY_S:
            moving.backward = false;
            break;
        case GLFW_KEY_A:
            moving.left = false;
            break;
        case GLFW_KEY_D:
            moving.right = false;
            break;
        case GLFW_KEY_LEFT:
            moving.turn_left = false;
            break;
        case GLFW_KEY_RIGHT:
            moving.turn_right = false;
            break;
        case GLFW_KEY_C:
            moving.up = false;
            break;
        case GLFW_KEY_V:
            moving.down = false;
            break;
        case GLFW_KEY_UP:
            moving.turn_up = false;
            break;
        case GLFW_KEY_DOWN:
            moving.turn_down = false;
            break;
        }
    }
}

class TracerWindow
{
  public:
    TracerWindow(CUstream stream,
        OptixPipeline pipeline,
        OptixShaderBindingTable sbt,
        Params params,
        int window_width = WINDOW_WIDTH,
        int window_height = WINDOW_HEIGHT,
        std::string outfile = "")
        : window_width(window_width), width(window_width - SIDE_PANEL_WIDTH), height(window_height),
          buf_width(width / DOWNSAMPLING), buf_height(height / DOWNSAMPLING), outfile(std::move(outfile)),
          output_buffer(sutil::CUDAOutputBufferType::CUDA_DEVICE, buf_width, buf_height), params(params),
          pipeline(pipeline), stream(stream), sbt(sbt)
    {
        //
        // Initialize GLFW state
        //
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
        buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
        buffer.width = buf_width;
        buffer.height = buf_height;
        buffer.data = output_buffer.getHostPointer();

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
    }

    void run()
    {

        GLDisplay display(buffer.pixel_format);
#ifdef NDEBUG
        spf = 5;
#else
        spf = 1;
#endif
        params.image = output_buffer.map();
        params.image_width = buf_width;
        params.image_height = buf_height;
        params.alloc_film();

        // gui

        /*
        buffer.data = output_buffer.getHostPointer();
        buffer.width = buf_width;
        buffer.height = buf_height;
        buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
        */


        int framebuf_res_x = 0, framebuf_res_y = 0;
        int step = 0;

        spdlog::info("start loop");
        while (glfwWindowShouldClose(window) == 0) {
            glfwPollEvents();
            // glfwWaitEvents();
            params.dirty = false;

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            ImGui::SetNextWindowPos({ 0, 0 }, ImGuiCond_::ImGuiCond_Always);
            ImGui::SetNextWindowSize(
                { static_cast<float>(SIDE_PANEL_WIDTH), static_cast<float>(height) }, ImGuiCond_Always);

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
            triangles->imgui();

            ImGui::End();
            if (render_to_file && !outfile.empty()) {
                render_to_file = false;

                sutil::saveImage(outfile.c_str(), buffer, false);
                spdlog::info("saving rendered image to {}", outfile);
                spdlog::info("total samples: {}", params.dt);
            }

            // render
            if (moving.forward) {
                cam.move_forward();
                params.dirty = true;
            }
            if (moving.backward) {
                cam.move_backward();
                params.dirty = true;
            }
            if (moving.left) {
                cam.move_left();
                params.dirty = true;
            }
            if (moving.right) {
                cam.move_right();
                params.dirty = true;
            }
            if (moving.turn_left) {
                cam.turn_left();
                params.dirty = true;
            }
            if (moving.turn_right) {
                cam.turn_right();
                params.dirty = true;
            }
            if (moving.up) {
                cam.move_up();
                params.dirty = true;
            }
            if (moving.down) {
                cam.move_down();
                params.dirty = true;
            }
            if (moving.turn_up) {
                cam.turn_up();
                params.dirty = true;
            }
            if (moving.turn_down) {
                cam.turn_down();
                params.dirty = true;
            }
            cam.set_fov(fov);
            cam.set_fd(fod);
            cam.set_ortho(orhto);
            cam.set_aperture(aperture);
            cam.compute_uvw();
            cam.update_device_ptr(params.camera);

            params.image = output_buffer.map();
            params.image_width = buf_width;
            params.image_height = buf_height;

            if (params.dirty) params.dt = 0;
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


            ImGui::Render();

            // cam.

            glfwGetFramebufferSize(window, &framebuf_res_x, &framebuf_res_y);
            display.display(buffer.width,
                buffer.height,
                SIDE_PANEL_WIDTH,
                0,
                framebuf_res_x - SIDE_PANEL_WIDTH,
                framebuf_res_y,
                pbo);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(window);
        }

        glfwDestroyWindow(window);
        glfwTerminate();
    }

  private:
    // window parameters
    int window_width;
    int width;
    int height;

    int buf_width = width / DOWNSAMPLING;
    int buf_height = height / DOWNSAMPLING;

    std::string outfile;

    float fov = 45.0f;
    float fod = 2.0f;
    int spf = 8;
    float aperture = 0.0f;
    bool orhto = false;
    GLFWwindow *window = nullptr;

    sutil::ImageBuffer buffer;
    sutil::CUDAOutputBuffer<uchar4> output_buffer;
    GLuint pbo = 0u;

    Camera cam;
    Params params;
    TriangleGAS *triangles = nullptr;
    Device device;
    OptixPipeline pipeline;
    CUstream stream;
    OptixShaderBindingTable sbt;
};
