#include "../shared/common.h"

// Penalizare miscare din diferenta de cadre
__global__ void computeMotionSalienceKernel(const unsigned char* __restrict__ d_curGray,
                                             const unsigned char* __restrict__ d_prevGray,
                                             float*               __restrict__ d_motion,
                                             int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float diff = fabsf((float)d_curGray[idx] - (float)d_prevGray[idx]);
    d_motion[idx] = diff * 5.0f;
}

// Initializare index coloana originala
__global__ void initOrigIdxKernel(int* d_origIdx, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    d_origIdx[y * width + x] = x;
}

// Mapeaza cusaturile eliminate inapoi la coordonate originale
__global__ void updateAccumMapKernel(const int* __restrict__ d_seamMap,
                                     const int* __restrict__ d_origIdx,
                                     int*       __restrict__ d_accumSeamMap,
                                     int curWidth, int origWidth, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= curWidth || y >= height) return;

    if (d_seamMap[y * curWidth + x] == 1) {
        int origX = d_origIdx[y * curWidth + x];
        d_accumSeamMap[y * origWidth + origX] = 1;
    }
}

// Bias energie spre coloanele taiate anterior (stabilitate temporala)
__global__ void accumMapBiasKernel(float*       __restrict__ d_energy,
                                   const float* __restrict__ d_motion,
                                   const int*   __restrict__ d_prevAccumMap,
                                   const int*   __restrict__ d_origIdx,
                                   int curWidth, int origWidth, int height,
                                   float bonus)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= curWidth || y >= height) return;

    int origX = d_origIdx[y * curWidth + x];
    if (d_prevAccumMap[y * origWidth + origX] == 1) {
        int idx = y * curWidth + x;
        float diff = d_motion[idx] / 5.0f;
        float lambda = 1.0f / (1.0f + diff);
        d_energy[idx] -= bonus * lambda;
    }
}
