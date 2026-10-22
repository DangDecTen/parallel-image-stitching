# Image Stitching

Panorama stitching combines overlapping photos into one wide image, as phone cameras do in panorama mode. The pipeline finds distinctive points in each photo, matches them, estimates the geometric transform between the photos, and warps and blends them together.



## Table of Contents

1. [Pipeline](#pipeline)
1. [Baseline](#basline-methods-and-parameters)
1. [How to run](#how-to-run)
1. [Development Stack](#development-stack)
1. [Project Plan](#project-plan)
1. [Repository Structure](#repository-structure)
1. [References](#references)


## Pipeline

```
Image A ─► Stage 1-3 (Image A) ─┐
                                ├─► Stage 4-6 ─► Panorama
Image B ─► Stage 1-3 (Image B) ─┘
```

Stages 1–3 run independently on each image, so both images can be processed in parallel or in separate CUDA streams. Stages 4–6 need data from both images.
1. Preprocessing
2. Keypoint detection
3. Description
4. Matching
5. Homography estimation
6. Warping and blending



## Basline methods and parameters

All parameters live in one header (`src/common.h`) that every version includes, so all versions use *identical* settings. Integer arithmetic is used wherever possible so CPU and GPU results can match exactly.

| Stage | Method | Why (quality) | Why (parallel efficiency) |
|---|---|---|---|
| 1 Preprocess | Integer grayscale; separable 5-tap binomial blur | BRIEF needs smoothing to be noise-robust | Separable: 2k instead of k² work per pixel; stencil → shared-memory tiling |
| 2 Keypoints | FAST-9 (t=20), 3×3 NMS, top-K=4000 | Cheap, repeatable corners | One thread per pixel; compaction + top-K are the interesting parallel patterns |
| 3 Description | BRIEF-256, patch ±15, fixed-seed pattern | Good enough for small rotations (we rotate, not walk) | 256-bit binary descriptors = 8×`uint32`; trivial per-keypoint threads |
| 4 Matching | Brute-force Hamming, ratio 0.8, max dist 80, cross-check | Ratio + cross-check remove ambiguous matches | All-pairs, like matmul: shared-memory tiling + `__popc` |
| 5 Homography | RANSAC, 4-point normalized DLT, 2000 iters, 3 px, refit on inliers | Robust to wrong matches | Hypotheses independent → one thread/warp per hypothesis; inlier count = reduction |
| 6 Warp/blend | Inverse warp, bilinear, distance feathering | No holes, hides seams | Gather per output pixel; texture memory for bilinear |



## How to run

### Prepare your dataset

Please organize your data using this layout.

```
data.zip
└── <site_name>/            e.g. indoor/, outdoor/, campus/, challenging_night/
    ├── small/  left.jpg  right.jpg      1280x720
    ├── medium/ left.jpg  right.jpg      1920x1080
    └── large/  left.jpg  right.jpg      3840x2160
```
Folders whose name starts with `challenging` are treated as the failure-analysis pair and excluded from the speed benchmark means.

### Run on Colab

Import the notebook (`notebooks/run_on_colab.ipynb`) on Colab, choose your GPU, and follow the notebook.



## Development Stack

| Need | Use | Why |
|-|-|-|
| Image I/O    | `stb_image.h` + `stb_image_write.h` (single headers)                                                        | No install on Colab, just download or commit them. Cite them in References.             |
| Build        | `nvcc` + a simple `Makefile` (or one `nvcc` command in a notebook cell)                                     | Compiles `.cpp` and `.cu` together                                                      |
| CPU timing   | `std::chrono::high_resolution_clock`                                                                        | Required for V0                                                                         |
| GPU timing   | CUDA events per stage                                                                                       | Required                                                                                |
| Profiling    | `nsys` (Nsight Systems) and `ncu` (Nsight Compute)                                                          | Required evidence per optimized version. Check both are available on the Colab runtime. |
| Verification | Your own comparison code (keypoint count, match count, homography difference, output image PSNR/diff vs V0) | “Consistent with V0” is a hard rule                                                     |



## Project Plan

| Stage | Requirements | Deliverable| Timeline|
|-|-|-|-|
| 01<br>(midterm) | - Selected pipeline (6 stages) and explain your decision<br>- Sequential version (V0) and Benchmarking<br>- Optimization roadmap | - Report (`.ipynb`)<br>- Code<br>- Dataset                                        | Week 1 — Scope, dataset, and method comparison<br>Week 2 — Build a working prototype and benchmark its stages<br>Week 3 — Freeze the method choices and defend the trade-offs |
| 02<br>(final)   | - Basic CUDA (V1)<br>- Optimized versions (V2, V3, etc.)<br>- Full benchmarking                                                  | - Report (`.ipynb`)<br>- Code<br>- Dataset<br>- Slides<br>- Video (20-30 minutes) | Week 4 — Complete V0 and V1<br>Week 5 — Implement V2 and V3<br>Week 6 — Benchmark, finalize, and rehearse                                                                     |



## Repository Structure

```text
root/
├── src/
│   ├── main.cpp          # CLI parsing, loading, timing loop, --version dispatch
│   ├── common.h          # Params, structs (Keypoint, Match, Homography), macros
│   ├── v0_seq.cpp        # sequential pipeline
│   ├── v1_cuda.cu        # basic CUDA
│   ├── v2_*.cu           # optimization 1
│   ├── v3_*.cu           # optimization 2
│   ├── verify.cpp        # compare outputs vs V0
│   └── stb_*.h
├── data/
├── notebooks/
├── Makefile
└── README.md
```


## References

1. E. Rosten, T. Drummond, "Machine learning for high-speed corner detection", ECCV 2006 (FAST).
2. M. Calonder et al., "BRIEF: Binary Robust Independent Elementary Features", ECCV 2010.
3. M. Fischler, R. Bolles, "Random sample consensus", CACM 1981 (RANSAC).
4. R. Hartley, A. Zisserman, *Multiple View Geometry in Computer Vision* (normalized DLT).
5. D. Lowe, "Distinctive image features from scale-invariant keypoints", IJCV 2004 (ratio test).
6. S. Nayar, "Image Stitching", FPCV-2-4 monograph (2025); R. Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed.
7. S. Barrett, `stb_image.h`, `stb_image_write.h` – https://github.com/nothings/stb (image I/O).
8. *(TODO: add every tutorial / repository / code snippet the team consulted, and cite it at the place of use.)*
