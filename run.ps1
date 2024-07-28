cd SDK
cmake --build --preset msvc-def --config Debug --target imgui_test && `
    ./builds/msvc/bin/Debug/imgui_test.exe `
        -m "C:/Users/andiw/3D Objects/bunny/reconstruction/bun_zipper.ply" `
        -o "last_render.png" `
        -w

cd ..
