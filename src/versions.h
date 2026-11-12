// versions.h - declarations shared by main.cpp and every version (v0_seq.cpp, v1_*.cu, ...): one run_vN() per version.
// Authors: <TODO: Ten - MSSV, ...>
// External sources: none (<TODO: sua neu co doan code lay/chinh tu nguon ngoai>)
#pragma once
#include "common.h"
// -1 = search the focal length automatically, 0 = planar (no projection), >0 = fixed focal length / image width.
extern double g_focal_frac;
// All versions share this signature. Returns false if stitching failed (e.g. too few inliers).
bool run_v0(const Image& A, const Image& B, Pano& out, StageTimes& t, Stats& s);
