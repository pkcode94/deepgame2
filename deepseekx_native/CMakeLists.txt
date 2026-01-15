cmake_minimum_required(VERSION 3.18)
project(deepseekx_native VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(deepseekx_native SHARED native_api.cpp native_api.h)

if (WIN32)
    set_target_properties(deepseekx_native PROPERTIES OUTPUT_NAME "deepseekx_native")
endif()

# Optionally link to libtorch (user must configure TORCH_DIR)
if(DEFINED ENV{TORCH_DIR})
    set(TORCH_DIR $ENV{TORCH_DIR})
    message(STATUS "TORCH_DIR=${TORCH_DIR}")
    # include and link against libtorch
    target_include_directories(deepseekx_native PRIVATE ${TORCH_DIR}/include)
    target_link_directories(deepseekx_native PRIVATE ${TORCH_DIR}/lib)

    # On Linux/macOS the lib name is libtorch.so / libtorch.dylib, on Windows it's torch.lib
    if(WIN32)
        find_library(LIBTORCH_LIB torch PATHS ${TORCH_DIR}/lib NO_DEFAULT_PATH)
        if(LIBTORCH_LIB)
            message(STATUS "Found libtorch: ${LIBTORCH_LIB}")
            target_link_libraries(deepseekx_native PRIVATE ${LIBTORCH_LIB})
            target_compile_definitions(deepseekx_native PRIVATE USE_LIBTORCH=1)
        else()
            message(WARNING "libtorch not found in ${TORCH_DIR}/lib")
        endif()
    else()
        # assume libtorch in lib directory
        target_link_libraries(deepseekx_native PRIVATE ${TORCH_DIR}/lib/libtorch.so)
        target_compile_definitions(deepseekx_native PRIVATE USE_LIBTORCH=1)
    endif()
else()
    message(STATUS "TORCH_DIR not set: building native stub (no libtorch)")
endif()
