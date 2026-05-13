// V2: JND-Based Video Seam Carving
// Implementare "Fast JND-Based Video Carving with GPU Acceleration" (Chiang et al.)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <string>
#include <iostream>
#include <algorithm>

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <ctime>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

namespace cg = cooperative_groups;

// Include kernel-uri
#include "kernels/shared/seam_extraction.cu"
#include "kernels/shared/pixel_removal.cu"
#include "kernels/shared/pixel_insertion.cu"
#include "kernels/v2/jnd_map.cu"
#include "kernels/v2/jnd_saliency.cu"
#include "kernels/v2/dp_forward.cu"
#include "kernels/shared/motion_salience.cu"

// Backtrack pe CPU
static void backtrackSeam(const int* h_backtrack, int width, int height,
                           int startCol, int* seamPath)
{
    // printf("backtrack: startCol=%d, w=%d, h=%d\n", startCol, width, height);
    int prev = startCol;
    for (int y = height - 1; y >= 0; --y) {
        seamPath[y] = prev;
        if (y == 0) break;

        int dir = h_backtrack[(y + 1) * width + prev];
        int nextCol = prev + dir;
        if (nextCol < 0) nextCol = 0;
        if (nextCol >= width) nextCol = width - 1;
        prev = nextCol;
    }
}

static void visualizeBatch(VideoWriter* vizWriter, Mat* vizScratch,
                           int* d_backtrack, int* d_seamCols, unsigned char* d_img,
                           int width, int height, int currentBatch, int origW,
                           int* h_backtrack, int* h_seamCols, int* h_seamPath,
                           int maxW, int maxH)
{
    if (!vizWriter || !vizWriter->isOpened()) return;

    CUDA_CHECK(cudaMemcpy(h_seamCols, d_seamCols, currentBatch * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_backtrack, d_backtrack, (size_t)width * height * sizeof(int), cudaMemcpyDeviceToHost));

    vizScratch->create(height, width, CV_8UC3);
    CUDA_CHECK(cudaMemcpy2D(vizScratch->data, vizScratch->step, d_img, width * 3, width * 3, height, cudaMemcpyDeviceToHost));

    for (int i = 0; i < currentBatch; ++i) {
        backtrackSeam(h_backtrack, width, height, h_seamCols[i], h_seamPath);
        for (int y = 0; y < height; ++y) {
            int sx = h_seamPath[y];
            if (sx >= 0 && sx < width)
                vizScratch->at<Vec3b>(y, sx) = Vec3b(0, 0, 255);
        }
        // Write the frame. To make it faster but still keep the video length correct,
        // we could write one frame per batch, but the user explicitly asked for the same speed as batch=1.
        // So we write 'currentBatch' frames.
        Mat padded = Mat::zeros(maxH, maxW, CV_8UC3);
        vizScratch->copyTo(padded(Rect(0, 0, width, height)));
        *vizWriter << padded;
    }
}

// Kernel-uri transpunere pt cusaturi orizontale
__global__ void transposeBGR(const unsigned char* in, unsigned char* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) {
        int inIdx = (y * w + x) * 3;
        int outIdx = (x * h + y) * 3;
        out[outIdx+0] = in[inIdx+0];
        out[outIdx+1] = in[inIdx+1];
        out[outIdx+2] = in[inIdx+2];
    }
}
__global__ void transposeGray(const unsigned char* in, unsigned char* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) out[x * h + y] = in[y * w + x];
}
__global__ void transposeFloat(const float* in, float* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) out[x * h + y] = in[y * w + x];
}
__global__ void transposeInt(const int* in, int* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) out[x * h + y] = in[y * w + x];
}

static void transposeState(unsigned char*& d_imgA, unsigned char*& d_imgB, 
                           unsigned char*& d_grayA, unsigned char*& d_grayB,
                           float*& d_modGrayA, float*& d_modGrayB,
                           float*& d_motionA, float*& d_motionB,
                           int*& d_origIdxA, int*& d_origIdxB,
                           int w, int h) {
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    transposeBGR<<<grid, block>>>(d_imgA, d_imgB, w, h);
    transposeGray<<<grid, block>>>(d_grayA, d_grayB, w, h);
    transposeFloat<<<grid, block>>>(d_modGrayA, d_modGrayB, w, h);
    transposeFloat<<<grid, block>>>(d_motionA, d_motionB, w, h);
    transposeInt<<<grid, block>>>(d_origIdxA, d_origIdxB, w, h);

    swap(d_imgA, d_imgB); swap(d_grayA, d_grayB);
    swap(d_modGrayA, d_modGrayB); swap(d_motionA, d_motionB);
    swap(d_origIdxA, d_origIdxB);
}

// Marcare cusatura pe GPU cu evitare coliziuni
__global__ void backtrackAndMarkSeamKernel(
    const int* d_backtrack,
    const int* d_seamCols,
    int* d_seamMap,
    int* d_masterSeamMap, 
    const int* d_origIdx, 
    int width, int height, int totalSeamsInBatch, int origW,
    bool isSim) 
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= totalSeamsInBatch) return;

    int curX = d_seamCols[s];
    if (curX < 0 || curX >= width) return;

    for (int y = height - 1; y >= 0; --y) {
        int targetX = curX;
        
        if (atomicCAS(&d_seamMap[y * width + targetX], 0, 1) != 0) {
            bool found = false;
            for (int dx = 1; dx < 10; ++dx) {
                int nx = targetX + dx;
                if (nx < width && atomicCAS(&d_seamMap[y * width + nx], 0, 1) == 0) {
                    targetX = nx; found = true; break;
                }
            }
            if (!found) {
                for (int dx = 1; dx < 10; ++dx) {
                    int nx = targetX - dx;
                    if (nx >= 0 && atomicCAS(&d_seamMap[y * width + nx], 0, 1) == 0) {
                        targetX = nx; found = true; break;
                    }
                }
            }
        }
        
        if (isSim) {
            int ox = d_origIdx[y * width + targetX];
            atomicAdd(&d_masterSeamMap[y * origW + ox], 1);
        }

        if (y > 0) {
            int dir = d_backtrack[y * width + curX];
            curX += dir;
            if (curX < 0) curX = 0;
            if (curX >= width) curX = width - 1;
        }
    }
}

// Logica principala de taiere pt un cadru
static void carveFrameJND(
    unsigned char*& d_imgA, unsigned char*& d_imgB,
    unsigned char*& d_grayA, unsigned char*& d_grayB,
    float*&         d_modGrayA, float*&         d_modGrayB,
    float*&         d_motionA, float*&         d_motionB,
    int*&           d_origIdxA, int*&           d_origIdxB,
    float*         d_jndSpatial, float*         d_jndTemporal, float*         d_jnd,
    float*         d_saliency, float*         d_cost,
    int*           d_accumSeamMap, int*           d_prevAccumSeamMap,
    int*           d_backtrack, int*           d_seamMap, int*           d_seamCols,
    int*           h_backtrack, int*           h_seamCols, int*           h_seamPath, int*           h_seamMap,
    int&           width, int            origW, int            height, int            totalSeams,
    int            maxW, int            maxH,
    float          lambda, float          energyBias, float          skinBias, float motionWeight,
    bool           isExpand, bool           hasPrevFrame,
    VideoWriter*   vizWriter = nullptr, Mat*           vizScratch = nullptr, int            batchSize = 1)
{
    if (isExpand) {
        int simWidth = width;
        int *d_masterSeamCount;
        
        size_t bytes3 = (size_t)maxW * height * 3;
        size_t bytes1 = (size_t)maxW * height;
        size_t bytesF = (size_t)maxW * height * sizeof(float);
        size_t bytesI = (size_t)maxW * height * sizeof(int);

        CUDA_CHECK(cudaMalloc(&d_masterSeamCount, bytesI));
        CUDA_CHECK(cudaMemset(d_masterSeamCount, 0, bytesI));

        unsigned char *d_imgSimA, *d_imgSimB, *d_graySimA, *d_graySimB;
        float *d_modGraySimA, *d_modGraySimB, *d_motionSimA, *d_motionSimB;
        int *d_origIdxSimA, *d_origIdxSimB;
        
        CUDA_CHECK(cudaMalloc(&d_imgSimA, bytes3)); CUDA_CHECK(cudaMalloc(&d_imgSimB, bytes3));
        CUDA_CHECK(cudaMalloc(&d_graySimA, bytes1)); CUDA_CHECK(cudaMalloc(&d_graySimB, bytes1));
        CUDA_CHECK(cudaMalloc(&d_modGraySimA, bytesF)); CUDA_CHECK(cudaMalloc(&d_modGraySimB, bytesF));
        CUDA_CHECK(cudaMalloc(&d_motionSimA, bytesF)); CUDA_CHECK(cudaMalloc(&d_motionSimB, bytesF));
        CUDA_CHECK(cudaMalloc(&d_origIdxSimA, bytesI)); CUDA_CHECK(cudaMalloc(&d_origIdxSimB, bytesI));
        

        CUDA_CHECK(cudaMemcpy(d_imgSimA, d_imgA, bytes3, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_graySimA, d_grayA, bytes1, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_modGraySimA, d_modGrayA, bytesF, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_motionSimA, d_motionA, bytesF, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_origIdxSimA, d_origIdxA, bytesI, cudaMemcpyDeviceToDevice));

        int seamsSimulated = 0;
        while (seamsSimulated < totalSeams) {
            int currentBatch = min(batchSize, totalSeams - seamsSimulated);
            dim3 block(BLOCK_DIM_2D, BLOCK_DIM_2D);
            dim3 grid((simWidth + block.x - 1) / block.x, (height + block.y - 1) / block.y);
            sobelSaliencyKernel<<<grid, block>>>(d_modGraySimA, d_graySimA, d_imgSimA, d_motionSimA, d_saliency, simWidth, height, skinBias, motionWeight);

            CUDA_CHECK(cudaMemcpy(d_cost, d_saliency, simWidth * sizeof(float), cudaMemcpyDeviceToDevice));
            launchDpAllRowsCoopV2(d_modGraySimA, d_saliency, d_cost, d_backtrack, simWidth, height);

            size_t shmem = simWidth * sizeof(float);
            int minSep = (currentBatch > 1) ? max(2, simWidth / (currentBatch + 5)) : 1;
            findMultiSeamBottomKernel<<<1, min(simWidth, 1024), shmem>>>(d_cost, d_seamCols, simWidth, height, currentBatch, minSep);
            
            CUDA_CHECK(cudaMemset(d_seamMap, 0, (size_t)simWidth * height * sizeof(int)));
            backtrackAndMarkSeamKernel<<<(currentBatch + 31) / 32, 32>>>(d_backtrack, d_seamCols, d_seamMap, d_masterSeamCount, d_origIdxSimA, simWidth, height, currentBatch, origW, true);
            if (vizWriter && vizWriter->isOpened()) {
                int curExpandedW = origW + seamsSimulated + currentBatch;
                // Generate the current expanded frame into d_imgB (which is safely unused during simulation)
                insertSeamsPrefixSumKernel<<<height, 256, (size_t)curExpandedW * sizeof(int)>>>(
                    d_imgA, d_imgB, d_grayA, d_grayB, d_motionA, d_motionB, d_origIdxA, d_origIdxB, 
                    d_masterSeamCount, origW, curExpandedW, height, true); // true = color inserted pixels red
                
                vizScratch->create(height, curExpandedW, CV_8UC3);
                CUDA_CHECK(cudaMemcpy2D(vizScratch->data, vizScratch->step, d_imgB, curExpandedW * 3, curExpandedW * 3, height, cudaMemcpyDeviceToHost));
                Mat padded = Mat::zeros(maxH, maxW, CV_8UC3);
                vizScratch->copyTo(padded(Rect(0, 0, curExpandedW, height)));
                for(int s=0; s<currentBatch; s++) *vizWriter << padded;
            }

            // We use d_masterSeamCount for the final expansion

            int nextSimW = simWidth - currentBatch;
            removeSeamsPrefixSumKernel<<<height, 256, (size_t)simWidth * sizeof(int)>>>(d_imgSimA, d_imgSimB, d_graySimA, d_graySimB, d_motionSimA, d_motionSimB, d_origIdxSimA, d_origIdxSimB, d_seamMap, simWidth, nextSimW, height);
            
            swap(d_imgSimA, d_imgSimB); swap(d_graySimA, d_graySimB);
            swap(d_modGraySimA, d_modGraySimB); swap(d_motionSimA, d_motionSimB);
            swap(d_origIdxSimA, d_origIdxSimB);
            
            simWidth = nextSimW;
            seamsSimulated += currentBatch;
            if (seamsSimulated < totalSeams) {
                modifiedIntensityKernel<<<grid, block>>>(d_graySimA, d_jnd, d_origIdxSimA, d_modGraySimA, simWidth, origW, height, lambda);
            }
        }

        // Final expansion step
        width = origW + totalSeams;
        insertSeamsPrefixSumKernel<<<height, 256, (size_t)width * sizeof(int)>>>(d_imgA, d_imgB, d_grayA, d_grayB, d_motionA, d_motionB, d_origIdxA, d_origIdxB, d_masterSeamCount, origW, width, height);
        swap(d_imgA, d_imgB); swap(d_grayA, d_grayB); swap(d_motionA, d_motionB); swap(d_origIdxA, d_origIdxB);


        cudaFree(d_imgSimA); cudaFree(d_imgSimB);
        cudaFree(d_graySimA); cudaFree(d_graySimB);
        cudaFree(d_modGraySimA); cudaFree(d_modGraySimB);
        cudaFree(d_motionSimA); cudaFree(d_motionSimB);
        cudaFree(d_origIdxSimA); cudaFree(d_origIdxSimB);
        cudaFree(d_masterSeamCount);
        return;
    }

    int seamsProcessed = 0;
    while (seamsProcessed < totalSeams) {
        int currentBatch = min(batchSize, totalSeams - seamsProcessed);
        dim3 block(BLOCK_DIM_2D, BLOCK_DIM_2D);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        sobelSaliencyKernel<<<grid, block>>>(d_modGrayA, d_grayA, d_imgA, d_motionA, d_saliency, width, height, skinBias, motionWeight);

        // Debug: dump saliency/energy map (once)
        {
            static bool dumpedSaliency = false;
            if (!dumpedSaliency) {
                dumpedSaliency = true;
                vector<float> h_sal(width * height);
                CUDA_CHECK(cudaMemcpy(h_sal.data(), d_saliency, (size_t)width * height * sizeof(float), cudaMemcpyDeviceToHost));
                float maxVal = *max_element(h_sal.begin(), h_sal.end());
                if (maxVal < 1e-6f) maxVal = 1.0f;
                Mat energyImg(height, width, CV_8UC1);
                for (int i = 0; i < width * height; i++)
                    energyImg.data[i] = (unsigned char)(min(h_sal[i] / maxVal, 1.0f) * 255.0f);
                imwrite("debug_v2_saliency.png", energyImg);
                fprintf(stderr, "[DEBUG] debug_v2_saliency.png\n");
            }
        }


        if (hasPrevFrame) {
            accumMapBiasKernel<<<grid, block>>>(d_saliency, d_motionA, d_prevAccumSeamMap, d_origIdxA, width, origW, height, energyBias);
        }

        CUDA_CHECK(cudaMemcpy(d_cost, d_saliency, width * sizeof(float), cudaMemcpyDeviceToDevice));
        launchDpAllRowsCoopV2(d_modGrayA, d_saliency, d_cost, d_backtrack, width, height);

        size_t shmem = width * sizeof(float);
        int minSep = (currentBatch > 1) ? max(2, width / (currentBatch + 5)) : 1;
        findMultiSeamBottomKernel<<<1, min(width, 1024), shmem>>>(d_cost, d_seamCols, width, height, currentBatch, minSep);

        CUDA_CHECK(cudaMemset(d_seamMap, 0, (size_t)width * height * sizeof(int)));
        backtrackAndMarkSeamKernel<<<(currentBatch + 31) / 32, 32>>>(d_backtrack, d_seamCols, d_seamMap, nullptr, nullptr, width, height, currentBatch, origW, false);

        if (vizWriter && vizWriter->isOpened()) {
            visualizeBatch(vizWriter, vizScratch, d_backtrack, d_seamCols, d_imgA, width, height, currentBatch, origW, h_backtrack, h_seamCols, h_seamPath, maxW, maxH);
        }

        updateAccumMapKernel<<<grid, block>>>(d_seamMap, d_origIdxA, d_accumSeamMap, width, origW, height);

        int nextW = width - currentBatch;
        removeSeamsPrefixSumKernel<<<height, 256, (size_t)width * sizeof(int)>>>(d_imgA, d_imgB, d_grayA, d_grayB, d_motionA, d_motionB, d_origIdxA, d_origIdxB, d_seamMap, width, nextW, height);

        swap(d_imgA, d_imgB); swap(d_grayA, d_grayB);
        swap(d_modGrayA, d_modGrayB); swap(d_motionA, d_motionB);
        swap(d_origIdxA, d_origIdxB);

        width = nextW;
        seamsProcessed += currentBatch;
        if (seamsProcessed < totalSeams) {
            modifiedIntensityKernel<<<grid, block>>>(d_grayA, d_jnd, d_origIdxA, d_modGrayA, width, origW, height, lambda);
        }
    }
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        cout << "Utilizare: v2_jnd <input> <v_seams> [h_seams] [--output <path>] [--visualize <path>]" << endl;
        return 1;
    }

    string inputPath = argv[1];
    int verSeams      = atoi(argv[2]);
    int horSeams      = 0;
    string outputOverride = "";
    string vizPath = "";

    float lambda = 0.4f;
    float energyBias = 30.0f;
    float skinBias = 1000.0f;
    float motionWeight = 5.0f;
    int batchSize = 1;
    bool debugDump = false;
    bool isExpand = false;

    for (int i = 3; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--output" && i + 1 < argc) outputOverride = argv[++i];
        else if (arg == "--visualize" && i + 1 < argc) vizPath = argv[++i];
        else if (arg == "--lambda" && i + 1 < argc) lambda = (float)atof(argv[++i]);
        else if (arg == "--bias" && i + 1 < argc) energyBias = (float)atof(argv[++i]);
        else if (arg == "--skin-bias" && i + 1 < argc) skinBias = (float)atof(argv[++i]);
        else if (arg == "--motion-weight" && i + 1 < argc) motionWeight = (float)atof(argv[++i]);
        else if (arg == "--batch" && i + 1 < argc) batchSize = atoi(argv[++i]);
        else if (arg == "--expand") isExpand = true;
        else if (arg == "--debug") debugDump = true;
        else if (horSeams == 0 && isdigit(arg[0])) horSeams = atoi(arg.c_str());
    }

    VideoCapture cap(inputPath);
    if (!cap.isOpened()) {
        cerr << "Eroare: nu pot deschide " << inputPath << endl;
        return 1;
    }

    int origW   = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int origH   = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    int nFrames = (int)cap.get(CAP_PROP_FRAME_COUNT);
    double fps  = cap.get(CAP_PROP_FPS);

    int finalW = isExpand ? (origW + verSeams) : (origW - verSeams);
    int finalH = isExpand ? (origH + horSeams) : (origH - horSeams);

    printf("[JND] %dx%d -> %dx%d (%s)\n", origW, origH, finalW, finalH, isExpand ? "Expand" : "Shrink");
    // printf("verSeams=%d horSeams=%d batch=%d lambda=%.2f\n", verSeams, horSeams, batchSize, lambda);

    string outPath = outputOverride.empty() ? "output/v2_result.mp4" : outputOverride;
    VideoWriter writer(outPath, VideoWriter::fourcc('m','p','4','v'), fps, Size(finalW, finalH));

    VideoWriter vizWriter; Mat vizFrame;
    if (!vizPath.empty()) {
        int maxW = max(origW, finalW);
        int maxH = max(origH, finalH);
        vizWriter.open(vizPath, VideoWriter::fourcc('m','p','4','v'), 60.0, Size(maxW, maxH), true);
    }

    size_t imgBytes3 = (size_t)max(origW, finalW) * max(origH, finalH) * 3;
    size_t imgBytes1 = (size_t)max(origW, finalW) * max(origH, finalH);
    size_t floatBytes = (size_t)max(origW, finalW) * max(origH, finalH) * sizeof(float);
    size_t intBytes   = (size_t)max(origW, finalW) * max(origH, finalH) * sizeof(int);

    unsigned char *d_imgA, *d_imgB, *d_grayA, *d_grayB, *d_prevGray, *d_origGray;
    float *d_modGrayA, *d_modGrayB, *d_motionA, *d_motionB, *d_jndSpatial, *d_jndTemporal, *d_jnd, *d_saliency, *d_cost;
    int *d_backtrack, *d_seamMap, *d_seamCols, *d_origIdxA, *d_origIdxB, *d_accumSeamMap, *d_prevAccumSeamMap;

    CUDA_CHECK(cudaMalloc(&d_imgA, imgBytes3)); CUDA_CHECK(cudaMalloc(&d_imgB, imgBytes3));
    CUDA_CHECK(cudaMalloc(&d_grayA, imgBytes1)); CUDA_CHECK(cudaMalloc(&d_grayB, imgBytes1));
    CUDA_CHECK(cudaMalloc(&d_prevGray, imgBytes1)); CUDA_CHECK(cudaMalloc(&d_origGray, imgBytes1));
    CUDA_CHECK(cudaMalloc(&d_modGrayA, floatBytes)); CUDA_CHECK(cudaMalloc(&d_modGrayB, floatBytes));
    CUDA_CHECK(cudaMalloc(&d_motionA, floatBytes)); CUDA_CHECK(cudaMalloc(&d_motionB, floatBytes));
    CUDA_CHECK(cudaMalloc(&d_jndSpatial, floatBytes)); CUDA_CHECK(cudaMalloc(&d_jndTemporal, floatBytes));
    CUDA_CHECK(cudaMalloc(&d_jnd, floatBytes)); CUDA_CHECK(cudaMalloc(&d_saliency, floatBytes));
    CUDA_CHECK(cudaMalloc(&d_cost, floatBytes)); CUDA_CHECK(cudaMalloc(&d_backtrack, intBytes));
    CUDA_CHECK(cudaMalloc(&d_seamMap, intBytes)); CUDA_CHECK(cudaMalloc(&d_seamCols, 256 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_origIdxA, intBytes)); CUDA_CHECK(cudaMalloc(&d_origIdxB, intBytes));
    CUDA_CHECK(cudaMalloc(&d_accumSeamMap, intBytes)); CUDA_CHECK(cudaMalloc(&d_prevAccumSeamMap, intBytes));
    CUDA_CHECK(cudaMemset(d_prevAccumSeamMap, 0, intBytes));

    int* h_backtrack = (int*)malloc((size_t)max(origW, finalW) * max(origH, finalH) * sizeof(int));
    int* h_seamCols  = (int*)malloc(256 * sizeof(int));
    int* h_seamPath  = (int*)malloc(max(origW, origH) * sizeof(int));
    int* h_seamMap   = (int*)malloc((size_t)max(origW, finalW) * max(origH, finalH) * sizeof(int));

    Mat frame, outFrame;
    cudaEvent_t tStart, tStop;
    cudaEventCreate(&tStart);
    cudaEventCreate(&tStop);

    for (int f = 0; f < nFrames; ++f) {
        cap >> frame;
        if (frame.empty()) break;
        cudaEventRecord(tStart);

        int curW = origW, curH = origH;
        if (!frame.isContinuous()) frame = frame.clone();
        CUDA_CHECK(cudaMemcpy(d_imgA, frame.data, (size_t)curW * curH * 3, cudaMemcpyHostToDevice));

        dim3 block(BLOCK_DIM_2D, BLOCK_DIM_2D);
        dim3 grid((curW + block.x - 1) / block.x, (curH + block.y - 1) / block.y);
        bgrToGrayKernel<<<grid, block>>>(d_imgA, d_grayA, curW, curH);
        spatialJndKernel<<<grid, block>>>(d_grayA, d_jndSpatial, curW, curH);

        if (f > 0) {
            temporalJndKernel<<<grid, block>>>(d_grayA, d_prevGray, d_jndTemporal, curW, curH);
            computeMotionSalienceKernel<<<grid, block>>>(d_grayA, d_prevGray, d_motionA, curW, curH);
        } else {
            temporalJndKernel<<<grid, block>>>(d_grayA, d_grayA, d_jndTemporal, curW, curH);
            CUDA_CHECK(cudaMemset(d_motionA, 0, floatBytes));
        }

        CUDA_CHECK(cudaMemset(d_accumSeamMap, 0, intBytes));
        initOrigIdxKernel<<<grid, block>>>(d_origIdxA, curW, curH);
        combinedJndKernel<<<grid, block>>>(d_jndSpatial, d_jndTemporal, d_jnd, curW, curH);
        modifiedIntensityKernel<<<grid, block>>>(d_grayA, d_jnd, d_origIdxA, d_modGrayA, curW, origW, curH, lambda);

        // Debug: dump all pipeline stages as images (first frame only)
        if (debugDump && f == 2) {
            size_t npix = (size_t)curW * curH;

            // 1. Grayscale
            {
                vector<unsigned char> h_buf(npix);
                CUDA_CHECK(cudaMemcpy(h_buf.data(), d_grayA, npix, cudaMemcpyDeviceToHost));
                Mat img(curH, curW, CV_8UC1, h_buf.data());
                imwrite("debug_v2_gray.png", img);
                fprintf(stderr, "[DEBUG] debug_v2_gray.png\n");
            }

            // 2. Spatial JND
            {
                vector<float> h_buf(npix);
                CUDA_CHECK(cudaMemcpy(h_buf.data(), d_jndSpatial, npix * sizeof(float), cudaMemcpyDeviceToHost));
                float mx = *max_element(h_buf.begin(), h_buf.end());
                if (mx < 1e-6f) mx = 1.0f;
                Mat img(curH, curW, CV_8UC1);
                for (size_t i = 0; i < npix; i++) img.data[i] = (uchar)(min(h_buf[i]/mx, 1.0f)*255);
                imwrite("debug_v2_jnd_spatial.png", img);
                fprintf(stderr, "[DEBUG] debug_v2_jnd_spatial.png\n");
            }

            // 3. Temporal JND
            {
                vector<float> h_buf(npix);
                CUDA_CHECK(cudaMemcpy(h_buf.data(), d_jndTemporal, npix * sizeof(float), cudaMemcpyDeviceToHost));
                float mx = *max_element(h_buf.begin(), h_buf.end());
                if (mx < 1e-6f) mx = 1.0f;
                Mat img(curH, curW, CV_8UC1);
                for (size_t i = 0; i < npix; i++) img.data[i] = (uchar)(min(h_buf[i]/mx, 1.0f)*255);
                imwrite("debug_v2_jnd_temporal.png", img);
                fprintf(stderr, "[DEBUG] debug_v2_jnd_temporal.png\n");
            }

            // 4. Combined JND
            {
                vector<float> h_buf(npix);
                CUDA_CHECK(cudaMemcpy(h_buf.data(), d_jnd, npix * sizeof(float), cudaMemcpyDeviceToHost));
                float mx = *max_element(h_buf.begin(), h_buf.end());
                if (mx < 1e-6f) mx = 1.0f;
                Mat img(curH, curW, CV_8UC1);
                for (size_t i = 0; i < npix; i++) img.data[i] = (uchar)(min(h_buf[i]/mx, 1.0f)*255);
                imwrite("debug_v2_jnd.png", img);
                fprintf(stderr, "[DEBUG] debug_v2_jnd.png\n");
            }

            // 5. Modified (flattened) grayscale
            {
                vector<float> h_buf(npix);
                CUDA_CHECK(cudaMemcpy(h_buf.data(), d_modGrayA, npix * sizeof(float), cudaMemcpyDeviceToHost));
                Mat img(curH, curW, CV_8UC1);
                for (size_t i = 0; i < npix; i++) img.data[i] = (uchar)min(255.0f, max(0.0f, h_buf[i]));
                imwrite("debug_v2_modgray.png", img);
                fprintf(stderr, "[DEBUG] debug_v2_modgray.png\n");
            }

            // 6. Diff mask: original vs flattened (red = changed)
            {
                vector<unsigned char> h_gray(npix);
                vector<float> h_mod(npix);
                CUDA_CHECK(cudaMemcpy(h_gray.data(), d_grayA, npix, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_mod.data(), d_modGrayA, npix * sizeof(float), cudaMemcpyDeviceToHost));
                Mat img(curH, curW, CV_8UC3);
                for (size_t i = 0; i < npix; i++) {
                    float diff = fabsf((float)h_gray[i] - h_mod[i]);
                    if (diff > 1.0f) {
                        img.data[i*3+0] = 0;
                        img.data[i*3+1] = 0;
                        img.data[i*3+2] = (uchar)min(255.0f, diff * 10.0f);
                    } else {
                        img.data[i*3+0] = h_gray[i];
                        img.data[i*3+1] = h_gray[i];
                        img.data[i*3+2] = h_gray[i];
                    }
                }
                imwrite("debug_v2_diff_mask.png", img);
                fprintf(stderr, "[DEBUG] debug_v2_diff_mask.png\n");
            }

            fprintf(stderr, "[DEBUG] All pipeline images saved.\n");
        }

        CUDA_CHECK(cudaMemcpy(d_origGray, d_grayA, (size_t)curW * curH, cudaMemcpyDeviceToDevice));

        if (verSeams > 0) {
            carveFrameJND(d_imgA, d_imgB, d_grayA, d_grayB, d_modGrayA, d_modGrayB, d_motionA, d_motionB, d_origIdxA, d_origIdxB, d_jndSpatial, d_jndTemporal, d_jnd, d_saliency, d_cost, d_accumSeamMap, d_prevAccumSeamMap, d_backtrack, d_seamMap, d_seamCols, h_backtrack, h_seamCols, h_seamPath, h_seamMap, curW, origW, curH, verSeams, max(origW, finalW), max(origH, finalH), lambda, energyBias, skinBias, motionWeight, isExpand, f > 0, vizWriter.isOpened() ? &vizWriter : nullptr, &vizFrame, batchSize);
        }
        if (horSeams > 0) {
            transposeState(d_imgA, d_imgB, d_grayA, d_grayB, d_modGrayA, d_modGrayB, d_motionA, d_motionB, d_origIdxA, d_origIdxB, curW, curH);
            carveFrameJND(d_imgA, d_imgB, d_grayA, d_grayB, d_modGrayA, d_modGrayB, d_motionA, d_motionB, d_origIdxA, d_origIdxB, d_jndSpatial, d_jndTemporal, d_jnd, d_saliency, d_cost, d_accumSeamMap, d_prevAccumSeamMap, d_backtrack, d_seamMap, d_seamCols, h_backtrack, h_seamCols, h_seamPath, h_seamMap, curW, origW, curH, horSeams, max(origW, finalW), max(origH, finalH), lambda, energyBias, skinBias, motionWeight, isExpand, false, nullptr, nullptr, batchSize);
            transposeState(d_imgA, d_imgB, d_grayA, d_grayB, d_modGrayA, d_modGrayB, d_motionA, d_motionB, d_origIdxA, d_origIdxB, curW, curH);
            int temp = curW; curW = curH; curH = temp; int tOrig = origW; origW = origH; origH = tOrig;
        }

        outFrame.create(curH, curW, CV_8UC3);
        CUDA_CHECK(cudaMemcpy2D(outFrame.data, outFrame.step, d_imgA, curW * 3, curW * 3, curH, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(d_prevGray, d_origGray, (size_t)origW * origH, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_prevAccumSeamMap, d_accumSeamMap, intBytes, cudaMemcpyDeviceToDevice));

        if (outFrame.cols != finalW || outFrame.rows != finalH) {
            Mat tmp; resize(outFrame, tmp, Size(finalW, finalH)); outFrame = tmp;
        }
        writer << outFrame;

        cudaEventRecord(tStop);
        cudaEventSynchronize(tStop);
        float ms = 0;
        cudaEventElapsedTime(&ms, tStart, tStop);
        printf("\rCadru %d/%d (%.1f FPS)", f + 1, nFrames, 1000.0f / ms);
        fflush(stdout);
    }

    printf("\nGata\n");
    cap.release(); writer.release(); if (vizWriter.isOpened()) vizWriter.release();
    cudaFree(d_imgA); cudaFree(d_imgB); cudaFree(d_grayA); cudaFree(d_grayB);
    cudaFree(d_prevGray); cudaFree(d_origGray); cudaFree(d_modGrayA); cudaFree(d_modGrayB);
    cudaFree(d_motionA); cudaFree(d_motionB); cudaFree(d_jndSpatial); cudaFree(d_jndTemporal);
    cudaFree(d_jnd); cudaFree(d_saliency); cudaFree(d_cost); cudaFree(d_backtrack);
    cudaFree(d_seamMap); cudaFree(d_seamCols); cudaFree(d_origIdxA); cudaFree(d_origIdxB);
    cudaFree(d_accumSeamMap); cudaFree(d_prevAccumSeamMap);
    free(h_backtrack); free(h_seamCols); free(h_seamPath); free(h_seamMap);
    cudaEventDestroy(tStart); cudaEventDestroy(tStop);
    return 0;
}
