cd SDK
cmake --build --preset msvc-def --config Debug --target imgui_test

if ($LastExitCode -ne 0) {
    Write-Host("compilation failed")
    exit
} else {
    ./builds/msvc/bin/Debug/imgui_test.exe `
        -m "C:/Users/andiw/cpp/threed/robot.nbt" `
        -o "last_render.png" `
        -w
}

cd ..
