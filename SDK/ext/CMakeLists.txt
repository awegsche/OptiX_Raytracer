include(FetchContent)

FetchContent_Declare(
    cli11_proj
    QUIET
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.4.2
)

FetchContent_MakeAvailable(cli11_proj)

FetchContent_Declare(
    nbt
    GIT_REPOSITORY https://github.com/awegsche/nbt.git
    GIT_TAG main
    )
FetchContent_MakeAvailable(nbt)
