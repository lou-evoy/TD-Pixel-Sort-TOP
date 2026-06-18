# Pixel Sort TOP — real-time GPU pixel sorting for TouchDesigner (CUDA)

A pixel sorting TOP that reorders the pixels along each scanline — row or column — by a per-pixel
key, entirely on the GPU. The sort style is chosen from a drop-down and shaped by a single Amount slider.

## Demo

<!-- screenshots / GIFs / video go here -->
*Coming soon.*

## Why this one

- **Single-pass.** The whole frame is sorted with one device-wide radix sort over a composite
  key rather than per-row passes, which is where the speed comes from versus typical CPU or
  per-line GPU implementations.
- **Wide hardware support.** Fat binary covering NVIDIA Turing through Blackwell (RTX 20–50).
- **Allocate-once, zero-copy.** Buffers are reused across frames and the texture is read and
  written in place — no per-frame allocations, no host/device transfers.

## Getting the node

The compiled plugin isn't distributed in this repo. Precompiled builds will be available to
supporters on **Patreon** *(link coming soon)*. To compile it yourself, read on.

## Build it yourself

**Requirements:** TouchDesigner 2025.32050 (TOP C++ API v12), CUDA Toolkit 12.8+ (validated on 13.3),
Visual Studio 2022/2026 (Desktop development with C++), CMake 3.24+, and an NVIDIA GPU (Turing / RTX 20 or newer).

The TD C++ SDK headers (`TOP_CPlusPlusBase.h`, `CPlusPlus_Common.h`) are not in this repo — they ship
inside TouchDesigner at `<TD install>/Samples/CPlusPlus/CudaTOP`, and `-DTD_SDK_DIR` must point there
(the default assumes a standard `C:/Program Files/Derivative` install).

Run the build from the **x64 Native Tools Command Prompt for VS** (Start menu); a normal PowerShell/cmd
won't have `cl`/`nvcc` on `PATH`:

```bat
cmake -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ^
      -DTD_SDK_DIR="C:/Program Files/Derivative/TouchDesigner/Samples/CPlusPlus/CudaTOP"
cmake --build build
```

This produces `build/PixelSortTOP.dll`. Copy it to `%USERPROFILE%\Documents\Derivative\Plugins\`
(or run `cmake --build build --target install_to_td` to copy it there in one step), restart
TouchDesigner, and add the node from **OP Create → Custom → "Pixel Sort"**.

**Older toolkits / GPUs:** the build targets `sm_75`–`sm_120` by default, and `sm_120` (Blackwell / RTX 50)
needs CUDA 12.8+. To target a different set, override `PS_CUDA_ARCHITECTURES`, e.g.
`-DPS_CUDA_ARCHITECTURES="75-real;86-real;89-real"`; run `nvcc --list-gpu-code` to see what your toolkit supports.
