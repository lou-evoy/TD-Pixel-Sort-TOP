// CUDA error-checking helpers
#ifndef PIXELSORT_CUDA_CHECK_H
#define PIXELSORT_CUDA_CHECK_H

#include "cuda_runtime.h"
#include <cstdio>

// on failure: stash message, return cudaError_t from enclosing fn
#define PS_CUDA_RETURN(expr, outErrPtr)                                          \
    do {                                                                         \
        cudaError_t ps_err__ = (expr);                                           \
        if (ps_err__ != cudaSuccess) {                                           \
            ps_setError((outErrPtr), #expr, ps_err__, __FILE__, __LINE__);       \
            return ps_err__;                                                     \
        }                                                                        \
    } while (0)

// catches launch-config errors only, not async errors
#define PS_CUDA_CHECK_LAUNCH(outErrPtr)                                          \
    PS_CUDA_RETURN(cudaGetLastError(), (outErrPtr))

// static single-message buffer; one execute() at a time on TD main thread
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
