#include "common.h"

// Eliminare rand
__global__ void removeSeamsKernel(const unsigned char* __restrict__ d_imgIn,
                                 unsigned char*       __restrict__ d_imgOut,
                                 const unsigned char* __restrict__ d_grayIn,
                                 unsigned char*       __restrict__ d_grayOut,
                                 const int*           __restrict__ d_seamMap,
                                 int widthIn, int widthOut, int height)
{
    int y = blockIdx.x;
    int tid = threadIdx.x;
    if (y >= height) return;

    for (int x = tid; x < widthIn; x += blockDim.x) {
        if (d_seamMap[y * widthIn + x]) continue;
        int dst = 0;
        for (int i = 0; i < x; ++i)
            dst += (d_seamMap[y * widthIn + i] == 0) ? 1 : 0;

        int srcIdx = (y * widthIn + x) * 3;
        int dstIdx = (y * widthOut + dst) * 3;
        d_imgOut[dstIdx+0] = d_imgIn[srcIdx+0];
        d_imgOut[dstIdx+1] = d_imgIn[srcIdx+1];
        d_imgOut[dstIdx+2] = d_imgIn[srcIdx+2];
        d_grayOut[y * widthOut + dst] = d_grayIn[y * widthIn + x];
    }
}


__global__ void removeSeamsPrefixSumKernel(
        const unsigned char* __restrict__ d_imgIn,
        unsigned char*       __restrict__ d_imgOut,
        const unsigned char* __restrict__ d_grayIn,
        unsigned char*       __restrict__ d_grayOut,
        const float*         __restrict__ d_motionIn,
        float*               __restrict__ d_motionOut,
        const int*           __restrict__ d_origIdxIn,
        int*                 __restrict__ d_origIdxOut,
        const int*           __restrict__ d_seamMap,
        int widthIn, int widthOut, int height)
{
    // Static shared memory for the row's prefix sum
    extern __shared__ int s_scan[];
    __shared__ int s_blockTotal;

    int y = blockIdx.x;
    int tid = threadIdx.x;
    if (y >= height) return;

    int totalAcrossChunks = 0;

    for (int base = 0; base < widthIn; base += blockDim.x) {
        int x = base + tid;
        int val = (x < widthIn && d_seamMap[y * widthIn + x] == 0) ? 1 : 0;

        // 1. Warp scan
        int scan = warpScan(val);

        // 2. Collect warp totals
        __shared__ int s_warps[32];
        int lane = tid % 32;
        int wid  = tid / 32;
        if (lane == 31) s_warps[wid] = scan;
        __syncthreads();

        // 3. Scan warp totals
        if (tid < 32) {
            int v = (tid < (blockDim.x / 32)) ? s_warps[tid] : 0;
            int vscan = warpScan(v);
            s_warps[tid] = vscan;
        }
        __syncthreads();

        // 4. Final index for this chunk
        int blockOffset = (wid > 0) ? s_warps[wid - 1] : 0;
        if (x < widthIn) {
            s_scan[x] = totalAcrossChunks + blockOffset + scan - val;
        }

        // 5. Update total for next chunk
        if (tid == blockDim.x - 1) {
            s_blockTotal = blockOffset + scan;
        }
        __syncthreads();
        totalAcrossChunks += s_blockTotal;
        __syncthreads();
    }

    __syncthreads();

    // Muta pixelii folosind s_scan calculat
    // printf("remove row %d: totalAcross=%d\n", y, totalAcrossChunks);
    for (int x = tid; x < widthIn; x += blockDim.x) {
        if (d_seamMap[y * widthIn + x]) continue;
        
        int dst = s_scan[x];
        if (dst >= widthOut) continue; // Safety

        int srcIdx = (y * widthIn + x) * 3;
        int dstIdx = (y * widthOut + dst) * 3;
        
        d_imgOut[dstIdx+0] = d_imgIn[srcIdx+0];
        d_imgOut[dstIdx+1] = d_imgIn[srcIdx+1];
        d_imgOut[dstIdx+2] = d_imgIn[srcIdx+2];
        
        d_grayOut[y * widthOut + dst] = d_grayIn[y * widthIn + x];
        d_motionOut[y * widthOut + dst] = d_motionIn[y * widthIn + x];
        d_origIdxOut[y * widthOut + dst] = d_origIdxIn[y * widthIn + x];
    }
}

// Conversie BGR la Grayscale
__global__ void bgrToGrayKernel(const unsigned char* __restrict__ d_bgr,
                                unsigned char*       __restrict__ d_gray,
                                int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx3 = (y * width + x) * 3;
    float lum = 0.114f * d_bgr[idx3+0] + 0.587f * d_bgr[idx3+1] + 0.299f * d_bgr[idx3+2];
    d_gray[y * width + x] = (unsigned char)fminf(255.0f, lum);
}
