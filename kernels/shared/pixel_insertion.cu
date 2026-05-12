#include "common.h"


// Largeste imaginea prin inserare cusaturi
__global__ void insertSeamsPrefixSumKernel(
        const unsigned char* __restrict__ d_imgIn,
        unsigned char*       __restrict__ d_imgOut,
        const unsigned char* __restrict__ d_grayIn,
        unsigned char*       __restrict__ d_grayOut,
        const float*         __restrict__ d_motionIn,
        float*               __restrict__ d_motionOut,
        const int*           __restrict__ d_origIdxIn,
        int*                 __restrict__ d_origIdxOut,
        const int*           __restrict__ d_seamCount,
        int widthIn, int widthOut, int height, bool colorInserted = false)
{
    extern __shared__ int s_scan[];
    __shared__ int s_blockTotal;

    int y = blockIdx.x;
    int tid = threadIdx.x;
    if (y >= height) return;

    int totalAcrossChunks = 0;

    for (int base = 0; base < widthIn; base += blockDim.x) {
        int x = base + tid;
        int val = (x < widthIn) ? (1 + d_seamCount[y * widthIn + x]) : 0;

        int scan = warpScan(val);

        __shared__ int s_warps[32];
        int lane = tid % 32;
        int wid  = tid / 32;
        if (lane == 31) s_warps[wid] = scan;
        __syncthreads();

        if (tid < 32) {
            int v = (tid < (blockDim.x / 32)) ? s_warps[tid] : 0;
            int vscan = warpScan(v);
            s_warps[tid] = vscan;
        }
        __syncthreads();

        int blockOffset = (wid > 0) ? s_warps[wid - 1] : 0;
        if (x < widthIn) {
            s_scan[x] = totalAcrossChunks + blockOffset + scan - val;
        }

        if (tid == blockDim.x - 1) {
            s_blockTotal = blockOffset + scan;
        }
        __syncthreads();
        totalAcrossChunks += s_blockTotal;
        __syncthreads();
    }

    __syncthreads();

    for (int x = tid; x < widthIn; x += blockDim.x) {
        int count = d_seamCount[y * widthIn + x];
        int dst = s_scan[x];
        if (dst >= widthOut) continue;
        
        int srcIdx = (y * widthIn + x) * 3;
        int nextX = (x + 1 < widthIn) ? x + 1 : x;
        int nextSrcIdx = (y * widthIn + nextX) * 3;

        // Copiaza originalul
        // printf("insert: y=%d x=%d dst=%d\n", y, x, dst);
        int dstIdx = (y * widthOut + dst) * 3;
        d_imgOut[dstIdx+0] = d_imgIn[srcIdx+0];
        d_imgOut[dstIdx+1] = d_imgIn[srcIdx+1];
        d_imgOut[dstIdx+2] = d_imgIn[srcIdx+2];
        d_grayOut[y * widthOut + dst] = d_grayIn[y * widthIn + x];
        d_motionOut[y * widthOut + dst] = d_motionIn[y * widthIn + x];
        d_origIdxOut[y * widthOut + dst] = d_origIdxIn[y * widthIn + x];

        // Insereaza 'count' pixeli interpolati
        for (int i = 1; i <= count; ++i) {
            int insIdx = (y * widthOut + dst + i) * 3;
            if (dst + i >= widthOut) break;

            if (colorInserted) {
                d_imgOut[insIdx+0] = 0;
                d_imgOut[insIdx+1] = 0;
                d_imgOut[insIdx+2] = 255;
            } else {
                d_imgOut[insIdx+0] = (unsigned char)(((int)d_imgIn[srcIdx+0] + (int)d_imgIn[nextSrcIdx+0]) >> 1);
                d_imgOut[insIdx+1] = (unsigned char)(((int)d_imgIn[srcIdx+1] + (int)d_imgIn[nextSrcIdx+1]) >> 1);
                d_imgOut[insIdx+2] = (unsigned char)(((int)d_imgIn[srcIdx+2] + (int)d_imgIn[nextSrcIdx+2]) >> 1);
            }

            d_grayOut[y * widthOut + dst + i] = (unsigned char)(((int)d_grayIn[y * widthIn + x] + (int)d_grayIn[y * widthIn + nextX]) >> 1);
            d_motionOut[y * widthOut + dst + i] = d_motionIn[y * widthIn + x];
            d_origIdxOut[y * widthOut + dst + i] = d_origIdxIn[y * widthIn + x];
        }
    }
}
