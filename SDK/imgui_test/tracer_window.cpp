#include "tracer_window.h"

bool render_to_file = false;
Moving moving{};

TracerWindow::TracerWindow(CUstream stream,
    OptixPipeline pipeline,
    OptixShaderBindingTable sbt,
    Params params,
    int window_width,
    int window_height,
    std::string outfile,
    TriangleGAS *triangles)
    : window_width(window_width), width(window_width - SIDE_PANEL_WIDTH), height(window_height),
      buf_width(width / DOWNSAMPLING), buf_height(height / DOWNSAMPLING), outfile(std::move(outfile)),
      output_buffer(sutil::CUDAOutputBufferType::CUDA_DEVICE, buf_width, buf_height), params(params),
      triangles(triangles), pipeline(pipeline), stream(stream), sbt(sbt)
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

void TracerWindow::run() noexcept
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

    int framebuf_res_x = 0, framebuf_res_y = 0;

    spdlog::info("start loop");
    while (glfwWindowShouldClose(window) == 0) {
        glfwPollEvents();
        // glfwWaitEvents();
        params.dirty = false;

        imgui();

        update_camera();

        params.image = output_buffer.map();
        params.image_width = buf_width;
        params.image_height = buf_height;

        if (params.dirty) params.dt = 0;
        params.frame_step();

        const CUdeviceptr d_param = params.to_device();

        OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt, buf_width, buf_height, /*depth=*/1));
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

        glfwGetFramebufferSize(window, &framebuf_res_x, &framebuf_res_y);
        display.display(
            buffer.width, buffer.height, SIDE_PANEL_WIDTH, 0, framebuf_res_x - SIDE_PANEL_WIDTH, framebuf_res_y, pbo);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    glfwDestroyWindow(window);
    glfwTerminate();
}
void TracerWindow::imgui() noexcept
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowPos({ 0, 0 }, ImGuiCond_::ImGuiCond_Always);
    ImGui::SetNextWindowSize({ static_cast<float>(SIDE_PANEL_WIDTH), static_cast<float>(height) }, ImGuiCond_Always);

#ifdef NDEBUG
    ImGui::Begin("Release", 0, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
#else
    ImGui::Begin("Debug", 0, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
#endif
    if (ImGui::CollapsingHeader("Window")) {
        ImGui::Text("Window Width: %d", window_width);
        ImGui::Text("Window Height: %d", height);

        ImGui::Text("Buffer Width: %d", width);
        ImGui::Text("Buffer Height: %d", height);

        ImGui::Text("Camera:");
        // ImGui::Text("(%.2f, %.2f, %.2f", params.cam_eye.x, params.cam_eye.y, params.cam_eye.z);
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
    if (triangles != nullptr) triangles->imgui();

    ImGui::End();
    if (render_to_file && !outfile.empty()) {
        render_to_file = false;

        sutil::saveImage(outfile.c_str(), buffer, false);
        spdlog::info("saving rendered image to {}", outfile);
        spdlog::info("total samples: {}", params.dt);
    }
}

void TracerWindow::update_camera() noexcept
{
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
};
void initGL()
{
    spdlog::debug("init GL");
    if (gladLoadGL() == 0) throw std::runtime_error("Failed to initialize GL");

    GL_CHECK(glClearColor(0.212f, 0.271f, 0.31f, 1.0f));
    GL_CHECK(glClear(GL_COLOR_BUFFER_BIT));
}

