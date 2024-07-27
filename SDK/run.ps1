#cmake --build --preset debug-msvc --target imgui_test && ./builds/debug-msvc/bin/Debug/imgui_test.exe
cmake --build --preset msvc-def --config Debug --target imgui_test && `
    ./builds/msvc/bin/Debug/imgui_test.exe `
        -m "C:/Users/andiw/3D Objects/bunny/reconstruction/bun_zipper.ply"
