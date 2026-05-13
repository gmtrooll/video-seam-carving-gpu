# Video Seam Carving (GPU)

Content-aware video resizing using JND-based seam carving, accelerated with CUDA.

Based on the paper: *"Fast JND-Based Video Carving With GPU Acceleration for Real-Time Video Retargeting"* (Chiang et al.)

📄 **[Full documentation (PDF)](docs/documentatie_proiect_AM.pdf)**

## What it does

Instead of cropping or scaling, seam carving removes the least noticeable vertical/horizontal pixel paths from each frame. A perceptual model (JND) determines which areas the human eye is least sensitive to, so the removed pixels are as invisible as possible.

## Requirements

- NVIDIA GPU with CUDA support (tested on sm_86)
- CUDA Toolkit
- OpenCV 4
- Python 3 + NiceGUI (for the GUI)
- Linux / WSL

## Build

```bash
make all
```

You might need to change `-arch=sm_86` in the Makefile to match your GPU.

## Usage

### Command line

```bash
# shrink by 100 vertical seams
./v2_jnd input/video.mp4 100 --output output/result.mp4

# shrink + visualize seams
./v2_jnd input/video.mp4 100 --output output/result.mp4 --visualize output/viz.mp4

# expand by 50 seams
./v2_jnd input/video.mp4 50 --expand --output output/expanded.mp4

# all options
./v2_jnd <input> <v_seams> [h_seams] [--output <path>] [--visualize <path>]
         [--lambda <float>] [--bias <float>] [--skin-bias <float>]
         [--motion-weight <float>] [--batch <int>] [--expand] [--debug]
```

### GUI

```bash
pip install -r requirements.txt
python gui.py
```

Opens at `http://localhost:8080`.

## Project structure

```
v2_jnd.cu                       main file, orchestrates the pipeline
kernels/
  shared/
    common.h                    constants, CUDA_CHECK macro
    pixel_removal.cu            BGR->gray, seam removal (prefix sum)
    pixel_insertion.cu          seam insertion (for expand mode)
    seam_extraction.cu          DP backtracking + seam finding
    motion_salience.cu          inter-frame motion detection
  v2/
    jnd_map.cu                  spatial/temporal JND computation
    jnd_saliency.cu             intensity flattening + Sobel energy
    dp_forward.cu               forward energy DP (cooperative groups)
gui.py                          NiceGUI web interface
```

## Pipeline (per frame)

1. BGR to grayscale
2. Spatial JND map (luminance adaptation + texture masking)
3. Temporal JND map (inter-frame differences)
4. Intensity flattening (smooth out imperceptible gradients)
5. Sobel energy (70% flattened + 30% original + skin/motion bias)
6. Forward energy DP (cooperative groups for GPU-wide row sync)
7. Backtrack + remove seam
8. Repeat for N seams

## Documentation

See `docs/documentatie_proiect_AM.pdf` for the full write-up (in Romanian).
