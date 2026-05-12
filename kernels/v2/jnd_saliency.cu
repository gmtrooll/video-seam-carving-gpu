#include "../shared/common.h"

// Citire pixeli
__device__ inline float getPixF(const float* data, int x, int y, int w, int h) {
    int cx = max(0, min(w - 1, x));
    int cy = max(0, min(h - 1, y));
    return data[cy * w + cx];
}
__device__ inline float getPixB(const unsigned char* data, int x, int y, int w, int h) {
    int cx = max(0, min(w - 1, x));
    int cy = max(0, min(h - 1, y));
    return (float)data[cy * w + cx];
}

// Aplatizare intensitate sub pragul JND
__global__ void modifiedIntensityKernel(const unsigned char* __restrict__ d_gray,
                                         const float*         __restrict__ d_jnd,
                                         const int*           __restrict__ d_origIdx,
                                         float*               __restrict__ d_modifiedGray,
                                         int curWidth, int origWidth, int height,
                                         float lambda)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= curWidth || y >= height) return;

    int idx = y * curWidth + x;
    float I = (float)d_gray[idx];

    // Nu aplati muchii puternice (medie 5x5)
    float sum = 0.0f;
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            sum += getPixB(d_gray, x + dx, y + dy, curWidth, height);
        }
    }
    float bg = sum / 25.0f;

    // Sobel pe original pt detectie muchii
    float r00 = getPixB(d_gray, x-1, y-1, curWidth, height), r20 = getPixB(d_gray, x+1, y-1, curWidth, height);
    float r01 = getPixB(d_gray, x-1, y,   curWidth, height), r21 = getPixB(d_gray, x+1, y,   curWidth, height);
    float r02 = getPixB(d_gray, x-1, y+1, curWidth, height), r22 = getPixB(d_gray, x+1, y+1, curWidth, height);
    float r10 = getPixB(d_gray, x,   y-1, curWidth, height), r12 = getPixB(d_gray, x,   y+1, curWidth, height);

    float rgx = -r00 + r20 - 2.0f*r01 + 2.0f*r21 - r02 + r22;
    float rgy = -r00 - 2.0f*r10 - r20 + r02 + 2.0f*r12 + r22;
    if (fabsf(rgx) + fabsf(rgy) > 120.0f) {
        d_modifiedGray[idx] = I;
        return;
    }

    int origX = d_origIdx[idx];
    float threshold = lambda * d_jnd[y * origWidth + origX];

    if (fabsf(I - bg) < threshold) d_modifiedGray[idx] = bg;
    else                           d_modifiedGray[idx] = I;
}

// Calcul salienta cu Sobel pe intensitate modificata + miscare
__global__ void sobelSaliencyKernel(const float*         __restrict__ d_modifiedGray,
                                     const unsigned char* __restrict__ d_gray,
                                     const unsigned char* __restrict__ d_bgr,
                                     const float*         __restrict__ d_motion,
                                     float*               __restrict__ d_saliency,
                                     int width, int height,
                                     float skinBias, float motionWeight)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // Energie pe imaginea aplatizata
    float s00 = getPixF(d_modifiedGray, x-1, y-1, width, height);
    float s10 = getPixF(d_modifiedGray, x,   y-1, width, height);
    float s20 = getPixF(d_modifiedGray, x+1, y-1, width, height);
    float s01 = getPixF(d_modifiedGray, x-1, y,   width, height);
    float s21 = getPixF(d_modifiedGray, x+1, y,   width, height);
    float s02 = getPixF(d_modifiedGray, x-1, y+1, width, height);
    float s12 = getPixF(d_modifiedGray, x,   y+1, width, height);
    float s22 = getPixF(d_modifiedGray, x+1, y+1, width, height);

    float gx = -s00 + s20 - 2.0f*s01 + 2.0f*s21 - s02 + s22;
    float gy = -s00 - 2.0f*s10 - s20 + s02 + 2.0f*s12 + s22;
    float pEnergy = fabsf(gx) + fabsf(gy);

    // Energie pe imaginea originala
    float r00 = getPixB(d_gray, x-1, y-1, width, height);
    float r10 = getPixB(d_gray, x,   y-1, width, height);
    float r20 = getPixB(d_gray, x+1, y-1, width, height);
    float r01 = getPixB(d_gray, x-1, y,   width, height);
    float r21 = getPixB(d_gray, x+1, y,   width, height);
    float r02 = getPixB(d_gray, x-1, y+1, width, height);
    float r12 = getPixB(d_gray, x,   y+1, width, height);
    float r22 = getPixB(d_gray, x+1, y+1, width, height);

    float rgx = -r00 + r20 - 2.0f*r01 + 2.0f*r21 - r02 + r22;
    float rgy = -r00 - 2.0f*r10 - r20 + r02 + 2.0f*r12 + r22;
    float sEnergy = fabsf(rgx) + fabsf(rgy);

    // Detectie piele
    // printf("sal[%d,%d]: pE=%.1f sE=%.1f\n", x, y, pEnergy, sEnergy);
    unsigned char b = d_bgr[idx * 3 + 0];
    unsigned char g = d_bgr[idx * 3 + 1];
    unsigned char r = d_bgr[idx * 3 + 2];
    bool isSkin = (r > 95 && g > 40 && b > 20 && 
                   fmaxf(r, fmaxf(g, b)) - fminf(r, fminf(g, b)) > 15 && 
                   fabsf((float)r - g) > 15 && r > g && r > b);

    float total = 0.7f * pEnergy + 0.3f * sEnergy;
    if (isSkin) total += skinBias;
    
    // Adauga energie miscare pt protectie obiecte in miscare
    total += d_motion[idx] * motionWeight;

    d_saliency[idx] = total;
}

// Vizualizare efect aplatizare JND
__global__ void visualizeFlatteningKernel(const unsigned char* __restrict__ d_gray,
                                           const float*         __restrict__ d_modGray,
                                           unsigned char*       __restrict__ d_vizImg,
                                           int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float diff = fabsf((float)d_gray[idx] - d_modGray[idx]);
    int bgrIdx = idx * 3;

    if (diff > 0.05f) {
        d_vizImg[bgrIdx + 0] = 0;
        d_vizImg[bgrIdx + 1] = 0;
        d_vizImg[bgrIdx + 2] = 255;
    } else {
        unsigned char g = d_gray[idx];
        d_vizImg[bgrIdx + 0] = g;
        d_vizImg[bgrIdx + 1] = g;
        d_vizImg[bgrIdx + 2] = g;
    }
}
