cmake `
    -S . `
    -B build

cmake `
    -S . `
    -B build-clang `
    -G "Ninja" `
    -DCMAKE_CXX_COMPILER=clang++ `
    -DCMAKE_C_COMPILER=clang `
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1

cp build-clang/compile_commands.json .
