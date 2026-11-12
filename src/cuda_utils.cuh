// cuda_utils.cuh - small CUDA helpers shared by all GPU versions (V1, V2, ...): error checking + GPU timer.
// Not included by the CPU-only V0, so the g++ build is unaffected.
// Authors: <TODO: Ten - MSSV, ...>
// External sources: none (<TODO: sua neu co doan code lay tu nguon ngoai>)
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Wrap every CUDA runtime call:  CUDA_CHECK(cudaMalloc(&p, n));
#define CUDA_CHECK(call)                                                                        \
    do {                                                                                        \
        cudaError_t err_ = (call);                                                              \
        if (err_ != cudaSuccess) {                                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            exit(EXIT_FAILURE);                                                                 \
        }                                                                                       \
    } while (0)

// Check a kernel launch (call right after <<<...>>>); synchronizes so execution errors show up here.
#define CUDA_CHECK_KERNEL()                \
    do {                                   \
        CUDA_CHECK(cudaGetLastError());    \
        CUDA_CHECK(cudaDeviceSynchronize());\
    } while (0)

// Integer ceil-division for grid sizes.
static inline int div_up(int a, int b) { return (a + b - 1) / b; }

// GPU timer based on CUDA events:  GpuTimer t; t.start(); kernel<<<..>>>(); float ms = t.stop();
struct GpuTimer {
    cudaEvent_t a, b;
    GpuTimer()  { CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b)); }
    ~GpuTimer() { cudaEventDestroy(a); cudaEventDestroy(b); }
    void  start() { CUDA_CHECK(cudaEventRecord(a)); }
    float stop()  { CUDA_CHECK(cudaEventRecord(b)); CUDA_CHECK(cudaEventSynchronize(b));
                    float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, a, b)); return ms; }
};
