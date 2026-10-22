#pragma once
#include "common.h"
// All versions share this signature. Returns false if stitching failed (e.g. too few inliers).
bool run_v0(const Image& A, const Image& B, Pano& out, StageTimes& t, Stats& s);
