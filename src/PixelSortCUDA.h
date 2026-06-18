// Pixel Sort — CUDA algorithm interface (plain C++17, no kernel syntax)
// pipeline: ingest -> run-start scan -> key build -> CUB radix sort -> gather+mix
#ifndef PIXELSORT_CUDA_H
#define PIXELSORT_CUDA_H

#include "cuda_runtime.h"
#include <cstdint>

namespace pixelsort {

// max image dim; bounded by composite-key field width (13 bits => 8192)
static constexpr int kMaxDim = 8192;

// per-pixel scalar for sort key / mask; values MUST match setupParameters() menu order
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
    Horizontal = 0,   // line = row, length = width
    Vertical   = 1,   // line = column, length = height
};

enum class Order : int32_t
{
    Ascending  = 0,
    Descending = 1,
};

// how Amount slider [0,1] shapes result; 0 => identity, 1 => full sort
// values MUST match setupParameters() Sort Mode menu order
//   Full        : whole line is one interval (Amount ignored)
//   RevealDark  : reveal by key, darkest first (Amount = coverage)
//   RevealBright: reveal by key, brightest first
//   Random      : random-length intervals, whole line at Amount=1
//   Edges       : intervals bounded by key gradients; Amount = sensitivity
//   Melt        : blend within-run order between position and key
enum class Mode : int32_t
{
    Full         = 0,
    RevealDark   = 1,
    RevealBright = 2,
    Random       = 3,
    Edges        = 4,
    Melt         = 5,
};

// per-frame params, gathered host-side before CUDA work
struct Params
{
    int32_t  width  = 0;
    int32_t  height = 0;

    Axis     axis            = Axis::Horizontal;
    Order    order           = Order::Ascending;
    Channel  sortKey         = Channel::Luminance;
    Mode     mode            = Mode::Full;

    // drives selected Mode; 0 => identity, 1 => full sort
    float    amount          = 1.0f;

    // Random mode seed; animate for evolving randomness
    float    seed            = 0.0f;

    // BGRA8 vs RGBA8; only affects R/G/B channel mapping for the key, bytes moved verbatim
    bool     bgra            = true;

    // copy input to output; our own bypass since the API hides the native flag
    bool     bypass          = false;
};

// owns device buffers + CUB temp; one per node, realloc only when pixel count grows
class PixelSorter
{
public:
    PixelSorter() = default;
    ~PixelSorter();

    PixelSorter(const PixelSorter&) = delete;
    PixelSorter& operator=(const PixelSorter&) = delete;

    // runs pipeline on 'stream'; inSurf==0 clears to transparent black
    // *outError set on failure, valid until next process()
    // call between begin/endCUDAOperations(); does NOT sync the stream
    cudaError_t process(cudaSurfaceObject_t inSurf,
                        cudaSurfaceObject_t outSurf,
                        const Params& p,
                        cudaStream_t stream,
                        const char** outError);

private:
    cudaError_t ensureCapacity(int width, int height, const char** outError);
    void        freeAll();

    // allocated capacity in pixels; 0 => nothing allocated
    int64_t                 myCapacity = 0;

    // per-pixel line-order scratch; payload is the packed color itself
    uint16_t*               mySortKey    = nullptr;   // [N] quantized sort key
    uint8_t*                mySortable   = nullptr;   // [N] 1 if in sort
    uint32_t*               myScan       = nullptr;   // [N] packed (pos|reset)

    unsigned long long*     myKeys       = nullptr;   // [N] composite keys (in)
    unsigned long long*     myKeysAlt    = nullptr;   // [N] (out)
    uint32_t*               myVals       = nullptr;   // [N] packed RGBA payload (in)
    uint32_t*               myValsAlt    = nullptr;   // [N] (out)

    void*                   myCubTemp    = nullptr;   // shared scan/sort temp
    size_t                  myCubTempBytes = 0;
};

} // namespace pixelsort

#endif // PIXELSORT_CUDA_H
