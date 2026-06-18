/* CUDA error-checking helpers for the Pixel Sort TOP.
 *
 * Kernels launch asynchronously, so most launch/runtime errors only surface
 * at a later synchronizing call. We therefore (a) check every synchronous CUDA
 * Runtime / CUB call inline, and (b) check cudaGetLastError() right after each
 * kernel launch to catch bad launch configurations.
 *
 * Nothing here aborts the process: the algorithm layer returns a cudaError_t and
 * a human-readable message string up to the TD glue, which puts the node into a
 * clean error state via getErrorString().
 */
#ifndef PIXELSORT_CUDA_CHECK_H
#define PIXELSORT_CUDA_CHECK_H

#include "cuda_runtime.h"
#include <cstdio>

// Evaluate a CUDA Runtime / CUB expression. On failure: stash the formatted
// message in (outErrPtr) if provided, optionally log to stderr in debug, and
// 'return' the cudaError_t from the enclosing function. Use only in functions
// that return cudaError_t.
#define PS_CUDA_RETURN(expr, outErrPtr)                                          \
    do {                                                                         \
        cudaError_t ps_err__ = (expr);                                           \
        if (ps_err__ != cudaSuccess) {                                           \
            ps_setError((outErrPtr), #expr, ps_err__, __FILE__, __LINE__);       \
            return ps_err__;                                                     \
        }                                                                        \
    } while (0)

// Check for a kernel launch error (async errors are not caught here; the launch
// configuration / no-CUDA-context type errors are). Cheap: a single call.
#define PS_CUDA_CHECK_LAUNCH(outErrPtr)                                          \
    PS_CUDA_RETURN(cudaGetLastError(), (outErrPtr))

// Thread-local-ish single-message buffer owned by the algorithm layer. We keep
// it static (one per process) because only one execute() runs at a time on the
// TD main thread, and the message is consumed immediately after process().
inline char* ps_errorBuffer()
{
    static char buf[512];
    return buf;
}

inline void ps_setError(const char** outErrPtr, const char* expr,
                        cudaError_t err, const char* file, int line)
{
    char* buf = ps_errorBuffer();
    snprintf(buf, 512, "CUDA error %d (%s) at %s:%d -> %s",
             (int)err, cudaGetErrorString(err), file, line, expr);
#ifdef _DEBUG
    fprintf(stderr, "[PixelSortTOP] %s\n", buf);
#endif
    if (outErrPtr)
        *outErrPtr = buf;
}

#endif // PIXELSORT_CUDA_CHECK_H
