#include "../shared/common.h"

// DP forward energy pt un singur rand
__global__ void dpRowKernelV2(const float* __restrict__ d_modGray,
                              const float* __restrict__ d_saliency,
                              float*       __restrict__ d_cost,
                              int*         __restrict__ d_backtrack,
                              int width, int height, int row)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) return;

    int idx     = row * width + x;
    int prevIdx = (row - 1) * width + x;

    // Calc costs for 3 directions
    float left  = (x > 0)         ? d_modGray[idx - 1] : d_modGray[idx];
    float right = (x < width - 1) ? d_modGray[idx + 1] : d_modGray[idx];
    float up_px = d_modGray[prevIdx];

    float cU = fabsf(left - right);
    float cL = cU + fabsf(up_px - left);
    float cR = cU + fabsf(up_px - right);

    float up      = d_cost[prevIdx]     + cU;
    float upLeft  = (x > 0)         ? d_cost[prevIdx - 1] + cL : INF_COST;
    float upRight = (x < width - 1) ? d_cost[prevIdx + 1] + cR : INF_COST;

    float best = up;
    int   dir  = 0;
    if (upLeft  < best) { best = upLeft;  dir = -1; }
    if (upRight < best) { best = upRight; dir =  1; }

    d_cost[idx]      = d_saliency[idx] + best;
    d_backtrack[idx] = dir;
}

// Kernel DP complet cu cooperative groups (sincronizare intre randuri)
__global__ void dpAllRowsKernelV2(const float* __restrict__ d_modGray,
                                   const float* __restrict__ d_saliency,
                                   float*       __restrict__ d_cost,
                                   int*         __restrict__ d_backtrack,
                                   int width, int height)
{
    cg::grid_group grid = cg::this_grid();

    // Cache costuri rand anterior in shared mem
    extern __shared__ float s_prevCost[]; 

    int tid = threadIdx.x;
    int base_x = (blockIdx.x * blockDim.x + tid) * 4;
    int block_offset = 1; 

    for (int row = 1; row < height; ++row) {
        int prevRowBase = (row - 1) * width;
        
        // Load block of 4 + halos
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int x = base_x + i;
            if (x < width) s_prevCost[tid * 4 + i + block_offset] = d_cost[prevRowBase + x];
            else           s_prevCost[tid * 4 + i + block_offset] = INF_COST;
        }

        if (tid == 0) {
            int x_l = base_x - 1;
            s_prevCost[0] = (x_l >= 0) ? d_cost[prevRowBase + x_l] : INF_COST;
        }
        if (tid == blockDim.x - 1) {
            int x_r = base_x + 4;
            s_prevCost[blockDim.x * 4 + 1] = (x_r < width) ? d_cost[prevRowBase + x_r] : INF_COST;
        }
        __syncthreads();

        // Solve current row
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int x = base_x + i;
            if (x >= width) continue;

            int idx     = row * width + x;
            int prevIdx = (row - 1) * width + x;

            float l_p = (x > 0)         ? d_modGray[idx - 1] : d_modGray[idx];
            float r_p = (x < width - 1) ? d_modGray[idx + 1] : d_modGray[idx];
            float u_p = d_modGray[prevIdx];

            float cU = fabsf(l_p - r_p);
            float cL = cU + fabsf(u_p - l_p);
            float cR = cU + fabsf(u_p - r_p);

            float c_up = s_prevCost[tid * 4 + i + block_offset] + cU;
            float c_l  = s_prevCost[tid * 4 + i + block_offset - 1] + cL;
            float c_r  = s_prevCost[tid * 4 + i + block_offset + 1] + cR;

            float best = c_up;
            int dir = 0;
            if (c_l < best) { best = c_l; dir = -1; }
            if (c_r < best) { best = c_r; dir = 1; }

            d_cost[idx]      = d_saliency[idx] + best;
            d_backtrack[idx] = dir;
        }

        grid.sync(); // sync whole gpu for next row
    }
}

// Helper lansare kernel cooperativ
static bool launchDpAllRowsCoopV2(const float* d_modGray,
                                   float* d_saliency,
                                   float* d_cost,
                                   int*   d_backtrack,
                                   int    width,
                                   int    height)
{
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    if (!prop.cooperativeLaunch) return false;

    int nBlocks = (width + 4 * BLOCK_DIM - 1) / (4 * BLOCK_DIM);
    dim3 block(BLOCK_DIM);
    dim3 grid(nBlocks);
    size_t shMem = (4 * BLOCK_DIM + 2) * sizeof(float);

    void* args[] = { &d_modGray, &d_saliency, &d_cost, &d_backtrack, &width, &height };
    if (cudaLaunchCooperativeKernel((void*)dpAllRowsKernelV2, grid, block, args, shMem) != cudaSuccess)
        return false;

    return true;
}
