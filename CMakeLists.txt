# =============================================================================
# Pixel Sort TOP — TouchDesigner Custom Operator (CUDA)
#
# Produces PixelSortTOP.dll, a CUDA-mode C++ TOP. Validated with:
#   - TouchDesigner 2025.32050 (TOP C++ API v12)
#   - CUDA Toolkit 13.3 (CUB/CCCL 3.3)
#   - Visual Studio 2022/2026 (MSVC 14.4x), CMake >= 3.24
#
# Configure example (from a "x64 Native Tools" / Developer prompt):
#   cmake -S . -B build -G "Ninja" ^
#         -DCMAKE_BUILD_TYPE=Release ^
#         -DTD_SDK_DIR="C:/Program Files/Derivative/TouchDesigner/Samples/CPlusPlus/CudaTOP"
#   cmake --build build --config Release
# =============================================================================
cmake_minimum_required(VERSION 3.24)

project(PixelSortTOP LANGUAGES CXX CUDA)

# ---- Options ---------------------------------------------------------------
# Path to a folder containing the TD SDK headers TOP_CPlusPlusBase.h and
# CPlusPlus_Common.h. Defaults to the CudaTOP sample shipped with TouchDesigner.
# IMPORTANT: keep these headers matched to YOUR installed TD build.
set(TD_SDK_DIR
    "C:/Program Files/Derivative/TouchDesigner/Samples/CPlusPlus/CudaTOP"
    CACHE PATH "Folder containing TOP_CPlusPlusBase.h and CPlusPlus_Common.h")

# nvcc 13.3 may flag very new MSVC toolsets as 'unsupported'. This passes the
# build through. Turn OFF and use a supported toolset (e.g. MSVC 14.4x) if you
# prefer a fully validated host compiler.
option(PS_ALLOW_UNSUPPORTED_COMPILER "Pass -allow-unsupported-compiler to nvcc" ON)

if(NOT EXISTS "${TD_SDK_DIR}/TOP_CPlusPlusBase.h")
    message(FATAL_ERROR
        "TOP_CPlusPlusBase.h not found in TD_SDK_DIR='${TD_SDK_DIR}'.\n"
        "Point -DTD_SDK_DIR=... at a TD CPlusPlus sample folder (e.g. .../Samples/CPlusPlus/CudaTOP).")
endif()

# ---- Language standards ----------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---- GPU architectures: fat binary Turing..Blackwell + forward-compat PTX --
# 75  Turing   (RTX 20 / Quadro RTX 6000)
# 86  Ampere   (RTX 30 / A6000 / A4000)
# 89  Ada      (RTX 40)
# 120 Blackwell consumer (RTX 50) — requires CUDA >= 12.8 (we use 13.3)
# 120-virtual emits compute_120 PTX so future archs JIT-compile.
# Adjust if your toolkit predates Blackwell support (`nvcc --list-gpu-code`).
# Applied as a TARGET property below so it always wins over CMake's auto-default.
set(PS_CUDA_ARCHITECTURES "75-real;86-real;89-real;120-real;120-virtual"
    CACHE STRING "GPU architectures to embed (CMake CUDA_ARCHITECTURES syntax)")

# ---- Target ----------------------------------------------------------------
add_library(PixelSortTOP SHARED
    src/PixelSortTOP.cpp
    src/PixelSortCUDA.cu
)

find_package(CUDAToolkit REQUIRED)

target_include_directories(PixelSortTOP PRIVATE
    "${TD_SDK_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    ${CUDAToolkit_INCLUDE_DIRS}   # so the host .cpp finds cuda_runtime.h (cl.exe)
)

# CUB/CCCL is normally on nvcc's default include path in CUDA 13; add the cccl
# root explicitly as a safety net if present.
if(EXISTS "${CUDAToolkit_TARGET_DIR}/include/cccl")
    target_include_directories(PixelSortTOP PRIVATE "${CUDAToolkit_TARGET_DIR}/include/cccl")
endif()

# Statically link the CUDA runtime into the plugin so it does not depend on /
# clash with the cudart64_12.dll that TouchDesigner ships in its bin folder.
target_link_libraries(PixelSortTOP PRIVATE CUDA::cudart_static)

set_target_properties(PixelSortTOP PROPERTIES
    PREFIX ""                       # no "lib" prefix
    OUTPUT_NAME "PixelSortTOP"
    CUDA_ARCHITECTURES "${PS_CUDA_ARCHITECTURES}"  # fat binary + PTX (overrides default)
    CUDA_SEPARABLE_COMPILATION OFF  # single TU device code; no device linking needed
    CUDA_RUNTIME_LIBRARY Static     # statically link cudart: avoids clashing with
)                                   # TD's bundled cudart64_12.dll at runtime.

# Match TD's dynamic MSVC C runtime (/MD, /MDd). This is the CMake default but we
# state it explicitly so host and device objects agree.
set_property(TARGET PixelSortTOP PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

target_compile_definitions(PixelSortTOP PRIVATE
    $<$<CONFIG:Debug>:_DEBUG>
    WIN32_LEAN_AND_MEAN     # NOMINMAX is already defined by CPlusPlus_Common.h
)

target_compile_options(PixelSortTOP PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    # CCCL/CUB (CUDA 13) requires MSVC's conforming preprocessor on the host side.
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
)
if(PS_ALLOW_UNSUPPORTED_COMPILER)
    target_compile_options(PixelSortTOP PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>)
endif()

# ---- Optional: auto-install to TD's per-user plugin folder -----------------
# cmake --build build --target install_to_td
if(DEFINED ENV{USERPROFILE})
    set(TD_PLUGIN_DIR "$ENV{USERPROFILE}/Documents/Derivative/Plugins")
    add_custom_target(install_to_td
        COMMAND ${CMAKE_COMMAND} -E make_directory "${TD_PLUGIN_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:PixelSortTOP>" "${TD_PLUGIN_DIR}/"
        DEPENDS PixelSortTOP
        COMMENT "Copying PixelSortTOP.dll to ${TD_PLUGIN_DIR}")
endif()
