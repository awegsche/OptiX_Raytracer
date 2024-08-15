cd SDK
cmake --build --preset msvc-def --config Release --target imgui_test && `
    ./builds/msvc/bin/Release/imgui_test.exe `
        -m "C:/Users/andiw/cpp/threed/robot.nbt" `
        -w
cd ..
        #-m "C:/Users/andiw/3D Objects/source/plaza01/plaza01_night.FBX" `
        #-m "C:/Users/andiw/3D Objects/bunny/reconstruction/bun_zipper.ply" `
        #-m "C:/Users/andiw/3D Objects/spider_robot/source/Apocalyptic Spider Robot.fbx" `
