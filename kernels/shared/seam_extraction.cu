#include "common.h"

// Gaseste coloanele de start pt cusaturi pe ultimul rand
__global__ void findMultiSeamBottomKernel(const float* __restrict__ d_cost,
                                          int*         __restrict__ d_seamCols,
                                          int width, int height,
                                          int numSeams, int minSep)
{
    extern __shared__ float s_row[];

    int tid = threadIdx.x;
    int lastRow = height - 1;

    for (int i = tid; i < width; i += blockDim.x)
        s_row[i] = d_cost[lastRow * width + i];
    __syncthreads();

    if (tid != 0) return;

    for (int s = 0; s < numSeams; ++s) {
        float minVal = INF_COST;
        int   minCol = -1;
        for (int x = 0; x < width; ++x) {
            if (s_row[x] < minVal) {
                minVal = s_row[x];
                minCol = x;
            }
        }
        d_seamCols[s] = minCol;

        // Blocheaza vecinii pt a evita suprapunerea
        if (minCol >= 0) {
            int lo = max(0, minCol - minSep);
            int hi = min(width - 1, minCol + minSep);
            for (int x = lo; x <= hi; ++x)
                s_row[x] = INF_COST;
        }
    }
}

// La fel ca mai sus dar adauga bias temporal
__global__ void findMultiSeamTemporalKernel(const float* __restrict__ d_cost,
                                            const int*   __restrict__ d_prevSeamCols,
                                            int*         __restrict__ d_seamCols,
                                            int width, int height,
                                            int numSeams, int minSep, int temporalWindow)
{
    extern __shared__ float s_row[];

    int tid = threadIdx.x;
    int lastRow = height - 1;

    for (int i = tid; i < width; i += blockDim.x)
        s_row[i] = d_cost[lastRow * width + i];
    __syncthreads();

    if (tid != 0) return;

    for (int s = 0; s < numSeams; ++s) {
        float minVal = INF_COST;
        int   minCol = -1;

        for (int x = 0; x < width; ++x) {
            float cost = s_row[x];

            if (d_prevSeamCols != nullptr && s < 1) {
                int prevCol = d_prevSeamCols[s];
                int dist = abs(x - prevCol);
                if (dist <= temporalWindow) {
                    cost *= (1.0f - 0.8f * (1.0f - (float)dist / temporalWindow));
                }
            }

            if (cost < minVal) {
                minVal = cost;
                minCol = x;
            }
        }
        d_seamCols[s] = minCol;

        if (minCol >= 0) {
            int lo = max(0, minCol - minSep);
            int hi = min(width - 1, minCol + minSep);
            for (int x = lo; x <= hi; ++x)
                s_row[x] = INF_COST;
        }
    }
}
