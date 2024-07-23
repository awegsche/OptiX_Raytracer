#cmake --build --preset debug-msvc --target imgui_test && ./builds/debug-msvc/bin/Debug/imgui_test.exe
cmake --build --preset msvc-def --config Release --target imgui_test && ./builds/msvc/bin/Release/imgui_test.exe
