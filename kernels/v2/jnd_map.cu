#include "../shared/common.h"

// Citire pixel cu clampare la margini
__device__ inline float readGray(const unsigned char* data, int x, int y, int w, int h) {
    int cx = max(0, min(w - 1, x));
    int cy = max(0, min(h - 1, y));
    return (float)data[cy * w + cx];
}

// JND Spatial: adaptare luminanta si mascare textura
__global__ void spatialJndKernel(const unsigned char* __restrict__ d_gray,
                                 float*               __restrict__ d_jndSpatial,
                                 int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // 1. Mean background (5x5)
    float sum = 0.0f;
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            sum += readGray(d_gray, x + dx, y + dy, width, height);
        }
    }
    float bg = sum / 25.0f;

    // 2. Luminance adaptation
    float f1;
    if (bg <= 127.0f) f1 = 17.0f * (1.0f - sqrtf(bg / 127.0f)) + 3.0f;
    else              f1 = (3.0f / 128.0f) * (bg - 127.0f) + 3.0f;

    // 3. Sobel texture masking (3x3)
    float p00 = readGray(d_gray, x-1, y-1, width, height), p10 = readGray(d_gray, x, y-1, width, height), p20 = readGray(d_gray, x+1, y-1, width, height);
    float p01 = readGray(d_gray, x-1, y,   width, height),                                          p21 = readGray(d_gray, x+1, y,   width, height);
    float p02 = readGray(d_gray, x-1, y+1, width, height), p12 = readGray(d_gray, x, y+1, width, height), p22 = readGray(d_gray, x+1, y+1, width, height);

    float g1 = fabsf(-p00 + p20 - 2.0f*p01 + 2.0f*p21 - p02 + p22);
    float g2 = fabsf(-p00 - 2.0f*p10 - p20 + p02 + 2.0f*p12 + p22);
    float g3 = fabsf(2.0f*p00 - 2.0f*p22 + p10 - p12 + p01 - p21);
    float g4 = fabsf(2.0f*p20 - 2.0f*p02 + p10 - p12 + p21 - p01);

    float textMag = fmaxf(g1, fmaxf(g2, fmaxf(g3, g4)));

    // 4. Combinare JND (selectie conditionala)
    const float T_TEXT = 20.0f;
    const float BETA = 0.15f; 
    const float MIN_JND = 3.0f;

    float jnd = f1;
    if (textMag > T_TEXT) {
        float Tt = BETA * textMag;
        if (Tt > f1) jnd = Tt + MIN_JND;
    }

    const float JND_CAP = 30.0f;
    d_jndSpatial[y * width + x] = fminf(jnd, JND_CAP);
}

// Mascare temporala JND (diferenta inter-frame)
__global__ void temporalJndKernel(const unsigned char* __restrict__ d_curGray,
                                  const unsigned char* __restrict__ d_prevGray,
                                  float*               __restrict__ d_jndTemporal,
                                  int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float ild_sum = 0.0f;
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            ild_sum += fabsf(readGray(d_curGray, x + dx, y + dy, width, height) - readGray(d_prevGray, x + dx, y + dy, width, height));
        }
    }
    float avg_ild = ild_sum / 25.0f;

    const float ALPHA = 0.03f;
    d_jndTemporal[y * width + x] = 1.0f + ALPHA * avg_ild;
}

// Combinare harti JND (produs)
__global__ void combinedJndKernel(const float* __restrict__ d_jndSpatial,
                                  const float* __restrict__ d_jndTemporal,
                                  float*       __restrict__ d_jnd,
                                  int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    d_jnd[idx] = d_jndSpatial[idx] * d_jndTemporal[idx];
}
