cd SDK
cmake --preset clang
cp builds/clang/compile_commands.json .

cmake --preset msvc

cd ..
