// Pixel Sort — CUDA implementation
// validated: CUDA 13.3.33, CUB 3.3 (CCCL)
//
// whole-frame composite-key radix sort == per-scanline sort within masked runs
// line-order index k: line = k/L, pos = k%L, L = line length, N = W*H
// same k<->(x,y) map for ingest and scatter, so they are symmetric
// per frame: ingest -> buildScan -> InclusiveScan -> buildKeys -> SortPairs -> gather

#include "PixelSortCUDA.h"
#include "CudaCheck.h"

#include "device_launch_parameters.h"

// CUB / CCCL. In CUDA 13 the include root is <toolkit>/include/cccl (set by CMake).
#include <cub/device/device_scan.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <algorithm>

namespace pixelsort {

// --------------------------- composite key layout ---------------------------
// uint64 key, most-significant field first so a single ascending sort yields the
// final layout. Each line has exactly L pixels, so sorting by lineIndex groups
// lines contiguously; within a line, runStart orders the disjoint blocks (each
// sortable run or pinned non-sortable singleton starts at a unique position), and
// sortField orders pixels inside a run, with posInLine as a stable tie-break.
//
//   [ lineIndex : 13 ][ blockStart : 13 ][ sortField : 16 ]
//    bit 41..29         bit 28..16         bit 15..0
//
// There is NO explicit tie-break field: CUB DeviceRadixSort::SortPairs is stable, so
// pixels with equal keys keep their input-array order. We fill the arrays in line
// order (index k = line*L + pos increasing with pos within a line), so equal keys
// stay in ascending position — exactly what an explicit tie-break would give, but
// for free. This trims the key from 55 to 42 bits => one fewer 8-bit radix pass.
//
// 13 bits => values up to 8192 for lineIndex / blockStart (==kMaxDim).
// 16 bits => 65536 quantization levels for the sort field.
static constexpr int BITS_KEY  = 16; static constexpr int SHIFT_KEY  = 0;
static constexpr int BITS_RUN  = 13; static constexpr int SHIFT_RUN  = 16;
static constexpr int BITS_LINE = 13; static constexpr int SHIFT_LINE = 29;
static constexpr int END_BIT   = SHIFT_LINE + BITS_LINE; // 42
static constexpr uint32_t KEY_QUANT_MAX = (1u << BITS_KEY) - 1u; // 65535

static_assert(SHIFT_KEY + BITS_KEY  <= SHIFT_RUN,  "key/run fields overlap");
static_assert(SHIFT_RUN + BITS_RUN  <= SHIFT_LINE, "run/line fields overlap");
static_assert(SHIFT_LINE + BITS_LINE <= 64,        "composite key exceeds 64 bits");
static_assert((1 << BITS_LINE) >= kMaxDim, "BITS_LINE too small for kMaxDim");
static_assert((1 << BITS_RUN)  >= kMaxDim, "BITS_RUN too small for kMaxDim");

// ------------------------------ scan element --------------------------------
// Run-start scan, packed into one uint32 (4 bytes, half the memory traffic of a
// struct): low 31 bits = posInLine of the most recent reset; bit 31 = reset flag.
static constexpr uint32_t SCAN_FLAG = 0x80000000u;
static __host__ __device__ __forceinline__ uint32_t scanPack(uint32_t pos, bool reset)
{
    return (reset ? SCAN_FLAG : 0u) | (pos & 0x7fffffffu);
}

// Associative (not commutative) combine: the rightmost reset in a range wins its
// value; the range is a reset-range if either side contains a reset. Because every
// line's first pixel is a reset, runs never carry across line boundaries.
struct RunStartOp
{
    __host__ __device__ __forceinline__
    uint32_t operator()(uint32_t a, uint32_t b) const
    {
        uint32_t value = (b & SCAN_FLAG) ? (b & 0x7fffffffu) : (a & 0x7fffffffu);
        return ((a | b) & SCAN_FLAG) | value;
    }
};

// ------------------------------ small helpers -------------------------------
static __host__ __device__ __forceinline__ int divUp(int a, int b)
{
    return (a + b - 1) / b;
}

// Map a line-order index k (or sorted rank r) to a pixel coordinate. Identical
// for ingest (k) and scatter (r) — that symmetry is what makes the gather correct.
static __device__ __forceinline__ void lineOrderToXY(
    int k, int width, int height, Axis axis, int& x, int& y)
{
    if (axis == Axis::Horizontal)
    {
        // L = width: line = row, pos = column
        y = k / width;
        x = k - y * width;
    }
    else
    {
        // L = height: line = column, pos = row
        x = k / height;
        y = k - x * height;
    }
}

static __device__ __forceinline__ float4 toRGBA(uchar4 c, bool bgra)
{
    float r, g, b, a;
    if (bgra) { b = c.x; g = c.y; r = c.z; a = c.w; }
    else      { r = c.x; g = c.y; b = c.z; a = c.w; }
    const float inv = 1.0f / 255.0f;
    return make_float4(r * inv, g * inv, b * inv, a * inv);
}

// Scalar in [0,1] for the selected channel. Hue is normalized to [0,1).
static __device__ __forceinline__ float channelValue(float4 c, Channel chan)
{
    switch (chan)
    {
        case Channel::Red:        return c.x;
        case Channel::Green:      return c.y;
        case Channel::Blue:       return c.z;
        case Channel::Alpha:      return c.w;
        case Channel::Luminance:  return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; // Rec.709
        case Channel::Value:      return fmaxf(c.x, fmaxf(c.y, c.z));
        case Channel::Saturation:
        {
            float mx = fmaxf(c.x, fmaxf(c.y, c.z));
            float mn = fminf(c.x, fminf(c.y, c.z));
            return (mx <= 0.0f) ? 0.0f : (mx - mn) / mx;
        }
        case Channel::Hue:
        default:
        {
            float mx = fmaxf(c.x, fmaxf(c.y, c.z));
            float mn = fminf(c.x, fminf(c.y, c.z));
            float d  = mx - mn;
            if (d < 1e-6f) return 0.0f;
            float h;
            if      (mx == c.x) h = fmodf((c.y - c.z) / d, 6.0f);
            else if (mx == c.y) h = (c.z - c.x) / d + 2.0f;
            else                h = (c.x - c.y) / d + 4.0f;
            h *= (1.0f / 6.0f);
            if (h < 0.0f) h += 1.0f;
            return h;
        }
    }
}

// Integer hash, bit-identical on host and device (used by Random intervals and the
// per-line phase of Waves so the CPU reference can reproduce them exactly).
static __host__ __device__ __forceinline__ uint32_t hashU32(uint32_t x)
{
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}
static __host__ __device__ __forceinline__ uint32_t hash2(uint32_t a, uint32_t b)
{
    return hashU32((a * 0x9e3779b9U) ^ hashU32(b + 0x165667b1U));
}

// --------------------------------- kernels ----------------------------------

// 1. Read every pixel into linear line-order memory, quantize its sort key, and
//    compute eligibility (threshold mask AND, for Reveal modes, a key cutoff).
__global__ void ingestKernel(
    cudaSurfaceObject_t inSurf, int width, int height, Axis axis, bool bgra,
    Channel sortKey, Mode mode, uint32_t revealThresh,
    uint32_t* __restrict__ vals, uint16_t* __restrict__ skey, uint8_t* __restrict__ sortable)
{
    int N = width * height;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < N; k += gridDim.x * blockDim.x)
    {
        int x, y;
        lineOrderToXY(k, width, height, axis, x, y);

        uchar4 c;
        surf2Dread(&c, inSurf, x * (int)sizeof(uchar4), y, cudaBoundaryModeZero);
        // Sort payload = the packed pixel itself, so the gather is a coalesced copy.
        vals[k] = (uint32_t)c.x | ((uint32_t)c.y << 8) | ((uint32_t)c.z << 16) | ((uint32_t)c.w << 24);

        float4 rgba = toRGBA(c, bgra);

        // Quantized sort key (ascending/raw); descending is applied later in buildKeys.
        float sv = fminf(fmaxf(channelValue(rgba, sortKey), 0.0f), 1.0f);
        uint32_t kq = (uint32_t)(sv * (float)KEY_QUANT_MAX + 0.5f);
        skey[k] = (uint16_t)kq;

        // Eligibility: everything sorts, except Reveal modes gate by the sort-key cutoff.
        // Non-eligible pixels stay visible and pinned to their position.
        bool s = true;
        if (mode == Mode::RevealDark)        s = (kq <= revealThresh);
        else if (mode == Mode::RevealBright) s = (kq >= (KEY_QUANT_MAX - revealThresh));

        sortable[k] = s ? 1u : 0u;
    }
}

// 2. Emit ScanElem per k. A "reset" starts a new interval; the inclusive scan then
//    propagates each interval's start position (blockStart). Resets happen at line
//    starts, at sortable/non-sortable boundaries (so non-sortable pixels stay pinned
//    as singletons and runs never cross), and — inside a sortable run — wherever the
//    selected Mode introduces an interval break.
__global__ void buildScanKernel(
    const uint8_t* __restrict__ sortable, const uint16_t* __restrict__ skey,
    int width, int height, Axis axis, Mode mode, uint32_t seedU32,
    uint32_t breakProbU32, uint32_t edgeThresh, uint32_t* __restrict__ scan)
{
    int N = width * height;
    int L = (axis == Axis::Horizontal) ? width : height;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < N; k += gridDim.x * blockDim.x)
    {
        int line = k / L;
        int pos  = k - line * L;
        bool prevSortable = (pos > 0) && (sortable[k - 1] != 0u);
        bool reset = (pos == 0) || (sortable[k] == 0u) || !prevSortable;

        if (!reset)  // strictly inside a sortable run: apply the mode's interval break
        {
            bool brk = false;
            switch (mode)
            {
                case Mode::Random:
                    // Each candidate position has a fixed random rank; a break occurs
                    // when rank < breakProb. breakProb only decreases as Amount rises,
                    // so intervals merge monotonically (no reshuffle / jitter). seedU32
                    // shifts the whole random pattern.
                    brk = hash2((uint32_t)line * 0x9e3779b9u + seedU32, (uint32_t)pos) < breakProbU32;
                    break;
                case Mode::Edges:
                {
                    uint32_t a = skey[k], b = skey[k - 1];
                    uint32_t ad = (a > b) ? (a - b) : (b - a);
                    brk = (ad > edgeThresh);
                    break;
                }
                default: break; // Full, Reveal*, Melt -> no intra-run breaks
            }
            reset = brk;
        }

        scan[k] = scanPack((uint32_t)pos, reset);
    }
}

// 4. Build the composite key. blockStart (the interval start) comes from the scanned
//    value, or is 0 for every pixel when no interval breaks are possible (in which
//    case the whole line is one run and 'scan' is not computed — scan == nullptr).
//    The payload (packed color) is already in 'vals' from ingest and is left untouched.
__global__ void buildKeysKernel(
    const uint16_t* __restrict__ skey, const uint32_t* __restrict__ scan,
    int width, int height, Axis axis, Order order, Mode mode, float amount,
    float posScale, unsigned long long* __restrict__ keys)
{
    int N = width * height;
    int L = (axis == Axis::Horizontal) ? width : height;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < N; k += gridDim.x * blockDim.x)
    {
        int line = k / L;
        int pos  = k - line * L;
        uint32_t blockStart = scan ? (scan[k] & 0x7fffffffu) : 0u;

        uint32_t sf = skey[k];
        if (order == Order::Descending) sf = KEY_QUANT_MAX - sf; // reverse within-interval only

        // Within-interval ordering field. Melt interpolates between original position
        // (amount=0 -> identity) and the sort key (amount=1 -> sorted); all other modes
        // order purely by the key (their "amount" already shaped the intervals/eligibility).
        uint32_t field = sf;
        if (mode == Mode::Melt)
        {
            uint32_t pq = (uint32_t)((float)pos * posScale + 0.5f);
            float blended = (1.0f - amount) * (float)pq + amount * (float)sf;
            uint32_t f = (uint32_t)(blended + 0.5f);
            field = (f > KEY_QUANT_MAX) ? KEY_QUANT_MAX : f;
        }

        keys[k] =
              ((unsigned long long)line       << SHIFT_LINE)
            | ((unsigned long long)blockStart << SHIFT_RUN)
            | ((unsigned long long)field      << SHIFT_KEY);
    }
}

// 6. Scatter.
//    Proof of mapping: keys sort ascending with lineIndex as the most-significant
//    field and every line holding exactly L pixels, so sorted ranks [line*L, line*L+L)
//    belong to 'line', in increasing within-line target position. Thus rank r lands
//    at line-order position r, i.e. coordinate lineOrderToXY(r). The payload is already
//    the moved pixel's packed color, so this is a coalesced read + a surface write.
__global__ void gatherKernel(
    const uint32_t* __restrict__ sortedVals,
    cudaSurfaceObject_t outSurf, int width, int height, Axis axis)
{
    int N = width * height;
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < N; r += gridDim.x * blockDim.x)
    {
        uint32_t moved = sortedVals[r];
        int x, y;
        lineOrderToXY(r, width, height, axis, x, y);
        surf2Dwrite(moved, outSurf, x * (int)sizeof(uchar4), y, cudaBoundaryModeZero);
    }
}

// Bypass: copy input straight to output, unaffected (native TOP bypass behavior).
__global__ void passthroughKernel(
    cudaSurfaceObject_t inSurf, cudaSurfaceObject_t outSurf, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    uchar4 c;
    surf2Dread(&c, inSurf, x * (int)sizeof(uchar4), y, cudaBoundaryModeClamp);
    surf2Dwrite(c, outSurf, x * (int)sizeof(uchar4), y, cudaBoundaryModeZero);
}

// Used when no input is connected: clear to transparent black.
__global__ void clearKernel(cudaSurfaceObject_t outSurf, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    uchar4 zero = make_uchar4(0, 0, 0, 0);
    surf2Dwrite(zero, outSurf, x * (int)sizeof(uchar4), y, cudaBoundaryModeZero);
}

// ------------------------------ PixelSorter ---------------------------------

PixelSorter::~PixelSorter()
{
    freeAll();
}

void PixelSorter::freeAll()
{
    cudaFree(mySortKey);   mySortKey   = nullptr;
    cudaFree(mySortable);  mySortable  = nullptr;
    cudaFree(myScan);      myScan      = nullptr;
    cudaFree(myKeys);      myKeys      = nullptr;
    cudaFree(myKeysAlt);   myKeysAlt   = nullptr;
    cudaFree(myVals);      myVals      = nullptr;
    cudaFree(myValsAlt);   myValsAlt   = nullptr;
    cudaFree(myCubTemp);   myCubTemp   = nullptr;
    myCubTempBytes = 0;
    myCapacity = 0;
}

cudaError_t PixelSorter::ensureCapacity(int width, int height, const char** outError)
{
    int64_t N = (int64_t)width * (int64_t)height;
    if (N <= myCapacity)
        return cudaSuccess;  // existing buffers are large enough; reuse (no per-frame malloc)

    // Resolution grew: reallocate everything once for the new size.
    freeAll();

    PS_CUDA_RETURN(cudaMalloc(&mySortKey,  N * sizeof(uint16_t)),           outError);
    PS_CUDA_RETURN(cudaMalloc(&mySortable, N * sizeof(uint8_t)),            outError);
    PS_CUDA_RETURN(cudaMalloc(&myScan,     N * sizeof(uint32_t)),           outError);
    PS_CUDA_RETURN(cudaMalloc(&myKeys,     N * sizeof(unsigned long long)), outError);
    PS_CUDA_RETURN(cudaMalloc(&myKeysAlt,  N * sizeof(unsigned long long)), outError);
    PS_CUDA_RETURN(cudaMalloc(&myVals,     N * sizeof(uint32_t)),           outError);
    PS_CUDA_RETURN(cudaMalloc(&myValsAlt,  N * sizeof(uint32_t)),           outError);

    // Query CUB temp-storage requirements for BOTH the scan and the sort at this
    // size (temp ptr == nullptr only computes the size), then allocate the max once.
    size_t scanBytes = 0, sortBytes = 0;
    PS_CUDA_RETURN(cub::DeviceScan::InclusiveScan(
        nullptr, scanBytes, myScan, myScan, RunStartOp(), (int)N), outError);
    PS_CUDA_RETURN(cub::DeviceRadixSort::SortPairs(
        nullptr, sortBytes, myKeys, myKeysAlt, myVals, myValsAlt,
        (int)N, 0, END_BIT), outError);

    myCubTempBytes = std::max(scanBytes, sortBytes);
    PS_CUDA_RETURN(cudaMalloc(&myCubTemp, myCubTempBytes), outError);

    myCapacity = N;
    return cudaSuccess;
}

cudaError_t PixelSorter::process(
    cudaSurfaceObject_t inSurf, cudaSurfaceObject_t outSurf,
    const Params& p, cudaStream_t stream, const char** outError)
{
    if (outError) *outError = nullptr;

    const int W = p.width, H = p.height;
    const int N = W * H;

    // No input: clear and bail (still a valid, defined output).
    if (!inSurf)
    {
        dim3 b(16, 16, 1);
        dim3 g(divUp(W, b.x), divUp(H, b.y), 1);
        clearKernel<<<g, b, 0, stream>>>(outSurf, W, H);
        PS_CUDA_CHECK_LAUNCH(outError);
        return cudaSuccess;
    }

    // Bypass: pass the input through untouched, like a native TOP's Bypass flag.
    if (p.bypass)
    {
        dim3 b(16, 16, 1);
        dim3 g(divUp(W, b.x), divUp(H, b.y), 1);
        passthroughKernel<<<g, b, 0, stream>>>(inSurf, outSurf, W, H);
        PS_CUDA_CHECK_LAUNCH(outError);
        return cudaSuccess;
    }

    PS_CUDA_RETURN(ensureCapacity(W, H, outError), outError);

    const int BS = 256;
    // Cap grid to a sane size; grid-stride loops in the kernels cover the rest.
    int grid = std::min(divUp(N, BS), 65535 * 4);

    // --- Map the single Amount slider to each mode's parameter (host side) ---
    const int   L        = (p.axis == Axis::Horizontal) ? W : H;
    const float amount   = p.amount;
    const float posScale = (L > 1) ? ((float)KEY_QUANT_MAX / (float)(L - 1)) : 0.0f; // Melt
    // Reveal / Edges: cutoff & gradient threshold in quantized key units.
    const uint32_t revealThresh = (uint32_t)(amount * (float)KEY_QUANT_MAX);
    const uint32_t edgeThresh   = (uint32_t)(amount * (float)KEY_QUANT_MAX);
    // Random: per-pixel break probability = (1-amount) * L^(-amount). This is 1 at
    // amount=0 (break every pixel -> identity) and exactly 0 at amount=1 (no breaks
    // -> fully sorted), and decreases monotonically, so raising Amount only merges
    // intervals (interval length grows geometrically toward the whole line).
    float pBreak = (1.0f - amount) * powf((float)(L > 1 ? L : 2), -amount);
    if (pBreak < 0.0f) pBreak = 0.0f;
    const uint32_t breakProbU32 = (uint32_t)(pBreak * 4294967295.0);
    const uint32_t seedU32 = hashU32((uint32_t)(p.seed * 1000.0f + 0.5f));

    // Intervals (hence a per-pixel blockStart) only arise from reveal eligibility or
    // the Random/Edges break rules. When none apply, every line is a single run
    // starting at 0, so we skip the scan entirely and pass a null scan to buildKeys
    // (blockStart == 0).
    const bool needScan =
           p.mode == Mode::RevealDark || p.mode == Mode::RevealBright
        || p.mode == Mode::Random     || p.mode == Mode::Edges;

    ingestKernel<<<grid, BS, 0, stream>>>(
        inSurf, W, H, p.axis, p.bgra,
        p.sortKey, p.mode, revealThresh,
        myVals, mySortKey, mySortable);
    PS_CUDA_CHECK_LAUNCH(outError);

    const uint32_t* scanForKeys = nullptr;
    if (needScan)
    {
        buildScanKernel<<<grid, BS, 0, stream>>>(
            mySortable, mySortKey, W, H, p.axis, p.mode, seedU32,
            breakProbU32, edgeThresh, myScan);
        PS_CUDA_CHECK_LAUNCH(outError);

        // In-place inclusive scan: propagate each interval's start position (blockStart).
        PS_CUDA_RETURN(cub::DeviceScan::InclusiveScan(
            myCubTemp, myCubTempBytes, myScan, myScan, RunStartOp(), N, stream), outError);

        scanForKeys = myScan;
    }

    buildKeysKernel<<<grid, BS, 0, stream>>>(
        mySortKey, scanForKeys, W, H, p.axis, p.order, p.mode, amount,
        posScale, myKeys);
    PS_CUDA_CHECK_LAUNCH(outError);

    PS_CUDA_RETURN(cub::DeviceRadixSort::SortPairs(
        myCubTemp, myCubTempBytes, myKeys, myKeysAlt, myVals, myValsAlt,
        N, 0, END_BIT, stream), outError);

    gatherKernel<<<grid, BS, 0, stream>>>(
        myValsAlt, outSurf, W, H, p.axis);
    PS_CUDA_CHECK_LAUNCH(outError);

    return cudaSuccess;
}

} // namespace pixelsort
