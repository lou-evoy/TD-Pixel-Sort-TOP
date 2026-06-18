# Pixel Sort TOP — real-time GPU pixel sorting for TouchDesigner (CUDA)

A custom pixel sorting effect TOP that reorders the pixels along each scanline — row or column — by a per-pixel
key, entirely on the GPU. Different sort morphing methods are selectable via a drop down menu and are driven by a single amount parameter.

## Demo

<!-- screenshots / GIFs / video go here -->
*Coming soon.*

## Why this one

- **Single-pass architecture.** The whole frame is sorted with one device-wide radix sort
  over a composite key, not per-row passes — the basis for its speed over typical CPU or
  per-line GPU implementations.
- **Wide hardware support.** Fat binary covering NVIDIA Turing through Blackwell (RTX 20–50).
- **Allocate-once, zero-copy.** Buffers are reused across frames and the texture is read and
  written in place — no per-frame allocations or host↔device transfers.

## Getting the node

The compiled plugin isn't distributed in this repo. Precompiled builds will be available to
supporters on **Patreon** *(link coming soon)*. If you'd rather compile it yourself, read on.

## Build it yourself

**Prerequisites**

- TouchDesigner 2025.30000+
- CUDA Toolkit 13.x (12.8+ for Blackwell / RTX 50)
- Visual Studio 2022 or 2026 (MSVC, *Desktop development with C++*)
- CMake ≥ 3.24

**Build (Release)**

From an *x64 Native Tools Command Prompt* (so `cl` and `nvcc` are on `PATH`):

```bat
cmake -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ^
      -DTD_SDK_DIR="C:/Program Files/Derivative/TouchDesigner/Samples/CPlusPlus/CudaTOP"
cmake --build build
```

Output: `build/PixelSortTOP.dll`. Copy it to `%USERPROFILE%\Documents\Derivative\Plugins\`,
restart TouchDesigner, and add the node from **OP Create → TOP → "Pixel Sort"**.
