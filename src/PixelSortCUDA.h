/* Pixel Sort — CUDA algorithm interface.
 *
 * This header is the only contract between the TouchDesigner glue (PixelSortTOP.cpp)
 * and the CUDA implementation (PixelSortCUDA.cu). It is plain C++17 with no CUDA
 * kernel syntax so it compiles under MSVC as well as nvcc.
 *
 * Pipeline (see PixelSortCUDA.cu for the proofs of correctness):
 *   ingest -> run-start segmented scan -> composite-key build -> CUB radix sort -> gather+mix
 *
 * Images are handled as CUDA surface objects (cudaArray), matching the real TD
 * CUDA-TOP API. We linearize the input into device memory in the ingest kernel,
 * sort in linear "line-order" space, then scatter back via a surface write.
 */
#ifndef PIXELSORT_CUDA_H
#define PIXELSORT_CUDA_H

#include "cuda_runtime.h"
#include <cstdint>

namespace pixelsort {

// Maximum supported size along either image dimension. Set by the composite-key
// field widths (13 bits => 8192). Covers 8K (7680x4320) with margin. See the
// key-layout block in PixelSortCUDA.cu before changing this.
static constexpr int kMaxDim = 8192;

// Selectable scalar derived from a pixel. Used both for the sort key and the
// threshold mask criterion. The integer values MUST match the menu item order
// declared in PixelSortTOP::setupParameters().
enum class Channel : int32_t
{
    Luminance = 0,
    Hue       = 1,
    Saturation= 2,
    Value     = 3,
    Red       = 4,
    Green     = 5,
    Blue      = 6,
    Alpha     = 7,
};

enum class Axis : int32_t
{
    Horizontal = 0,   // sort within each row;    line length = width
    Vertical   = 1,   // sort within each column; line length = height
};

enum class Order : int32_t
{
    Ascending  = 0,
    Descending = 1,
};

// Sort "style" — how the single Sort Amount slider [0,1] shapes the result.
// All map 0 => identity (no sorting) and 1 => full sort. Integer values MUST match
// the Sort Mode menu order in PixelSortTOP::setupParameters().
//   Full        : whole line is one interval (Amount ignored). satyarth "none".
//   RevealDark  : reveal by sort key, darkest/lowest first (Amount = coverage).
//   RevealBright: reveal by sort key, brightest/highest first.
//   Random      : random-length intervals that grow to the whole line at Amount=1. satyarth "random".
//   Edges       : intervals bounded by key gradients; Amount = edge sensitivity. satyarth "edges".
//   Melt        : interpolate within-run order between original position and key (Amount = blend).
enum class Mode : int32_t
{
    Full         = 0,
    RevealDark   = 1,
    RevealBright = 2,
    Random       = 3,
    Edges        = 4,
    Melt         = 5,
};

// All per-frame parameters, gathered on the host before any CUDA work begins.
struct Params
{
    int32_t  width  = 0;
    int32_t  height = 0;

    Axis     axis            = Axis::Horizontal;
    Order    order           = Order::Ascending;
    Channel  sortKey         = Channel::Luminance;
    Mode     mode            = Mode::Full;

    // Single slider that drives the selected Mode. 0 => identity, 1 => full sort.
    float    amount          = 1.0f;

    // Random-pattern seed (Random mode only). Change it for a different random
    // interval layout; animate it for evolving randomness.
    float    seed            = 0.0f;

    // True when the input/output cudaArray is BGRA8 (TD's preferred 8-bit format),
    // false for RGBA8. Only affects which surface channel maps to R/G/B when
    // deriving the key; the pixel bytes themselves are moved verbatim.
    bool     bgra            = true;

    // When true, copy the input straight to the output (no sort) — mirrors a native
    // TOP's Bypass flag, which the C++ TOP API does not expose to the plugin.
    bool     bypass          = false;
};

// Owns all device buffers and CUB temp storage. One instance per TOP node.
// Allocation happens lazily in process() and is only redone when the pixel count
// grows (resolution change) — never per frame, satisfying the real-time constraint.
class PixelSorter
{
public:
    PixelSorter() = default;
    ~PixelSorter();

    PixelSorter(const PixelSorter&) = delete;
    PixelSorter& operator=(const PixelSorter&) = delete;

    // Runs the full pipeline on 'stream'. 'inSurf' may be 0 (no input) in which
    // case the output is cleared to transparent black. Returns cudaSuccess on
    // success; on failure returns the error code and sets *outError to a static
    // message string valid until the next process() call.
    //
    // Must be called between TOP_Context::beginCUDAOperations()/endCUDAOperations().
    // Does NOT synchronize the stream (TD manages CUDA<->Vulkan ordering).
    cudaError_t process(cudaSurfaceObject_t inSurf,
                        cudaSurfaceObject_t outSurf,
                        const Params& p,
                        cudaStream_t stream,
                        const char** outError);

private:
    cudaError_t ensureCapacity(int width, int height, const char** outError);
    void        freeAll();

    // Capacity currently allocated for (in pixels). 0 => nothing allocated yet.
    int64_t                 myCapacity = 0;

    // Per-pixel scratch in line-order (see .cu). The sort payload is the packed
    // pixel color itself, so the gather is a coalesced copy and no separate pixel
    // buffer is needed.
    uint16_t*               mySortKey    = nullptr;   // [N] quantized sort-key value (reveal/edges/sort)
    uint8_t*                mySortable   = nullptr;   // [N] 1 if pixel participates in sort
    uint32_t*               myScan       = nullptr;   // [N] packed (pos|reset) in-place scan buffer

    unsigned long long*     myKeys       = nullptr;   // [N] composite keys (in)
    unsigned long long*     myKeysAlt    = nullptr;   // [N] (out)
    uint32_t*               myVals       = nullptr;   // [N] payload = packed RGBA color (in)
    uint32_t*               myValsAlt    = nullptr;   // [N] (out)

    void*                   myCubTemp    = nullptr;   // shared scan/sort temp storage
    size_t                  myCubTempBytes = 0;
};

} // namespace pixelsort

#endif // PIXELSORT_CUDA_H
