#ifndef __TRACER_WINDOW_H__
#define __TRACER_WINDOW_H__

#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#include <spdlog/spdlog.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Trackball.h>

#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include <optix_stubs.h>
#include <utility>

#include "camera.h"
#include "device.h"
#include "optixTriangle.h"
#include "optix_types.h"
#include "sutil/CUDAOutputBuffer.h"
#include "sutil/sutil.h"
#include "triangle_gas.h"

using sutil::GLDisplay;

// ---- CONSTS -------------------------------------------------------------------------------------
constexpr int32_t SIDE_PANEL_WIDTH = 256;
constexpr int WINDOW_WIDTH = 1024;
constexpr int WINDOW_HEIGHT = 768;

#ifdef NDEBUG
constexpr int DOWNSAMPLING = 1;
#else
constexpr int DOWNSAMPLING = 4;
#endif


// ---- Moving Struct ------------------------------------------------------------------------------
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

// ---- Global Variables ---------------------------------------------------------------------------
extern bool render_to_file;
extern Moving moving;

// ---- OpenGL Init and Callbacks ------------------------------------------------------------------
void initGL();

static void errorCallback(int error, const char *description)
{
    spdlog::error("GLFW Error {}: {}", error, description);
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

// ---- TracerWindow -------------------------------------------------------------------------------
class TracerWindow
{
  public:
    /**
     * @brief A Window that displays a perspective view into the current scene and showcases the
     * rendering progress in real time.
     *
     * @param stream
     * @param pipeline
     * @param sbt
     * @param params
     * @param window_width
     * @param window_height
     * @param outfile
     * @param triangles
     */
    TracerWindow(CUstream stream,
        OptixPipeline pipeline,
        OptixShaderBindingTable sbt,
        Params params,
        int window_width = WINDOW_WIDTH,
        int window_height = WINDOW_HEIGHT,
        std::string outfile = "",
        TriangleGAS *triangles = nullptr);

    void run() noexcept;

    void set_outfile(std::string const &outfile) noexcept { this->outfile = outfile; }

  private:
    // ---- Private Functions ----------------------------------------------------------------------
    void update_camera() noexcept;

    void imgui() noexcept;

    // ---- Fields ---------------------------------------------------------------------------------
    // ---- --- Window Parameters ------------------------------------------------------------------
    int window_width;
    int width;
    int height;

    int buf_width = width / DOWNSAMPLING;
    int buf_height = height / DOWNSAMPLING;

    std::string outfile;

    // ---- --- Local Camera Parameters ------------------------------------------------------------
    float fov = 45.0f;
    float fod = 2.0f;
    int spf = 8;
    float aperture = 0.0f;
    bool orhto = false;
    GLFWwindow *window = nullptr;

    // ---- --- Buffers and GL Interop -------------------------------------------------------------
    sutil::ImageBuffer buffer;
    sutil::CUDAOutputBuffer<uchar4> output_buffer;
    GLuint pbo = 0u;

    // ---- --- Render Objects ---------------------------------------------------------------------
    Camera cam;
    Params params;
    TriangleGAS *triangles = nullptr;
    Device device;
    OptixPipeline pipeline;
    CUstream stream;
    OptixShaderBindingTable sbt;
};

#endif
