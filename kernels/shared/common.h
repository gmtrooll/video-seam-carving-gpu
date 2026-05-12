#ifndef KERNEL_COMMON_H
#define KERNEL_COMMON_H

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>

namespace cg = cooperative_groups;

// macro verificare erori cuda
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Warp-level scan (prefix sum in registre)
__device__ inline int warpScan(int val) {
    for (int offset = 1; offset < 32; offset <<= 1) {
        int n = __shfl_up_sync(0xffffffff, val, offset);
        if (threadIdx.x % 32 >= offset) val += n;
    }
    return val;
}

// limite la compilare
constexpr int MAX_WIDTH         = 3840;   
constexpr int MAX_HEIGHT        = 2160;
constexpr int BLOCK_DIM         = 256;    
constexpr int BLOCK_DIM_2D      = 16;     
constexpr int MAX_BATCH         = 1;      
constexpr int MIN_SEP           = 2;      
constexpr float INF_COST        = 1e18f;  

#endif