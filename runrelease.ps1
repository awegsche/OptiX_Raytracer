cd SDK
cmake --build --preset msvc-def --config Release --target imgui_test && `
    ./builds/msvc/bin/Release/imgui_test.exe `
        -m "C:/Users/andiw/3D Objects/bunny/reconstruction/bun_zipper.ply" `
        -w
cd ..
        #-m "C:/Users/andiw/3D Objects/spider_robot/source/Apocalyptic Spider Robot.fbx" `
        #-m "C:/Users/andiw/3D Objects/source/plaza01/plaza01_night.FBX" `
