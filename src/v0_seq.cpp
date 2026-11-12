// v0_seq.cpp - V0: single-thread sequential reference implementation.
// Pipeline: gray -> blur -> FAST-9 + NMS + top-K -> BRIEF-256 -> brute-force Hamming match
//           -> RANSAC (4-pt DLT) + inlier refinement -> inverse warp + feather blend
// Authors: <TODO: Ten - MSSV, ...>
// External sources: none (<TODO: sua neu co doan code lay/chinh tu nguon ngoai>)
#include "versions.h"
#include <chrono>
#include <algorithm>
#include <climits>

using Clock = std::chrono::high_resolution_clock;
static inline double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }
static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// ---- Stage 1a: RGB -> gray, integer weights (77,150,29)/256
static void to_gray(const Image& im, std::vector<uint8_t>& g) {
    g.resize((size_t)im.w * im.h);
    for (size_t i = 0; i < g.size(); i++) {
        int r = im.rgb[3*i], gg = im.rgb[3*i+1], b = im.rgb[3*i+2];
        g[i] = (uint8_t)((77*r + 150*gg + 29*b + 128) >> 8);
    }
}

// ---- Stage 1b: separable 5-tap binomial blur, clamp-to-edge. Integer math => bit-exact on GPU too.
static void gauss_blur(const std::vector<uint8_t>& src, int w, int h, std::vector<uint8_t>& dst) {
    std::vector<uint16_t> tmp((size_t)w * h);
    dst.resize((size_t)w * h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            int s = 0;
            for (int k = -2; k <= 2; k++) s += P::BLUR_W[k+2] * src[(size_t)y*w + clampi(x+k, 0, w-1)];
            tmp[(size_t)y*w + x] = (uint16_t)s;                 // max 255*16
        }
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            int s = 0;
            for (int k = -2; k <= 2; k++) s += P::BLUR_W[k+2] * tmp[(size_t)clampi(y+k, 0, h-1)*w + x];
            dst[(size_t)y*w + x] = (uint8_t)((s + 128) >> 8);   // total weight 256
        }
}

// ---- Stage 2: FAST-9 score, 3x3 non-max suppression, top-K
// score = largest t such that 9 contiguous ring pixels are all brighter than c+t (or all darker than c-t);
// a pixel is a corner if score > FAST_T. Integer-only => identical on every platform.
static void detect_fast(const std::vector<uint8_t>& img, int w, int h, std::vector<Keypoint>& kps) {
    std::vector<int> score((size_t)w * h, 0);
    const int B = P::BORDER;
    for (int y = B; y < h - B; y++)
        for (int x = B; x < w - B; x++) {
            int c = img[(size_t)y*w + x], d[16];
            for (int k = 0; k < 16; k++) d[k] = img[(size_t)(y + FAST_RING[k][1])*w + x + FAST_RING[k][0]] - c;
            // high-speed pre-test (exact): any 9-arc contains >= 2 of the 4 compass pixels {0,4,8,12},
            // and they must share polarity beyond the threshold. Otherwise the score is <= FAST_T anyway.
            int nb = 0, nd = 0;
            for (int k = 0; k < 16; k += 4) { nb += d[k] > P::FAST_T; nd += d[k] < -P::FAST_T; }
            if (nb < 2 && nd < 2) continue;
            int best = 0;
            for (int s = 0; s < 16; s++) {
                int mb = INT_MAX, md = INT_MAX;
                for (int k = 0; k < 9; k++) { int v = d[(s+k) & 15]; mb = std::min(mb, v); md = std::min(md, -v); }
                best = std::max(best, std::max(mb, md));
            }
            if (best > P::FAST_T) score[(size_t)y*w + x] = best;
        }
    kps.clear();
    for (int y = B; y < h - B; y++)
        for (int x = B; x < w - B; x++) {
            int s = score[(size_t)y*w + x];
            if (s == 0) continue;
            bool keep = true;
            for (int dy = -1; dy <= 1 && keep; dy++)
                for (int dx = -1; dx <= 1; dx++) {
                    if (!dx && !dy) continue;
                    int ns = score[(size_t)(y+dy)*w + x + dx];
                    bool earlier = (dy < 0) || (dy == 0 && dx < 0);       // tie-break: earlier raster pixel wins
                    if (ns > s || (ns == s && earlier)) { keep = false; break; }
                }
            if (keep) kps.push_back({x, y, s});
        }
    // canonical order: score desc, then y, then x  (makes top-K deterministic on every version)
    std::sort(kps.begin(), kps.end(), [](const Keypoint& a, const Keypoint& b) {
        if (a.score != b.score) return a.score > b.score;
        if (a.y != b.y) return a.y < b.y;
        return a.x < b.x; });
    if ((int)kps.size() > P::TOPK) kps.resize(P::TOPK);
}

// ---- Stage 3: oriented BRIEF-256 (bit = I(p1) < I(p2) on the blurred image, pattern rotated to the keypoint orientation)
static void brief_describe(const std::vector<uint8_t>& img, int w, const std::vector<Keypoint>& kps,
                           std::vector<Descriptor>& desc) {
    std::vector<int> pat; make_brief_patterns(pat);
    int dir[P::ORI_BINS][2]; make_ori_dirs(dir);
    desc.assign(kps.size(), Descriptor());
    for (size_t i = 0; i < kps.size(); i++) {
        int x = kps[i].x, y = kps[i].y;
        const int* p = &pat[(size_t)keypoint_orientation(img.data(), w, x, y, dir) * P::BRIEF_BITS * 4];
        for (int b = 0; b < P::BRIEF_BITS; b++) {
            int v1 = img[(size_t)(y + p[4*b+1])*w + x + p[4*b+0]];
            int v2 = img[(size_t)(y + p[4*b+3])*w + x + p[4*b+2]];
            if (v1 < v2) desc[i].w[b >> 5] |= 1u << (b & 31);
        }
    }
}

// ---- Stage 4: brute-force Hamming, Lowe ratio test (best < 0.8*second) + cross-check
static void match_bf(const std::vector<Descriptor>& dA, const std::vector<Descriptor>& dB, std::vector<Match>& out) {
    int nA = (int)dA.size(), nB = (int)dB.size();
    std::vector<int> bestA(nA, INT_MAX), secA(nA, INT_MAX), idxA(nA, -1), bestB(nB, INT_MAX), idxB(nB, -1);
    for (int i = 0; i < nA; i++)
        for (int j = 0; j < nB; j++) {
            int d = 0;
            for (int k = 0; k < P::BRIEF_WORDS; k++) d += __builtin_popcount(dA[i].w[k] ^ dB[j].w[k]);
            if (d < bestA[i]) { secA[i] = bestA[i]; bestA[i] = d; idxA[i] = j; }   // ties: lowest j wins
            else if (d < secA[i]) secA[i] = d;
            if (d < bestB[j]) { bestB[j] = d; idxB[j] = i; }                       // ties: lowest i wins
        }
    out.clear();
    for (int i = 0; i < nA; i++) {
        if (idxA[i] < 0 || bestA[i] > P::MAX_HAMMING) continue;
        if (secA[i] != INT_MAX && !(bestA[i] * P::RATIO_DEN < secA[i] * P::RATIO_NUM)) continue;
        if (idxB[idxA[i]] != i) continue;                                          // cross-check
        out.push_back({i, idxA[i], bestA[i]});
    }
}

// ---- Stage 5: RANSAC (pre-generated samples, 4-pt DLT) + least-squares refinement on inliers
static bool ransac_homography(const std::vector<Keypoint>& kA, const std::vector<Keypoint>& kB,
                              const std::vector<Match>& m, double H[9], int& nin, double thr) {
    int n = (int)m.size();
    if (n < 4) return false;
    std::vector<float> src(2*n), dst(2*n);                // src = B points, dst = A points
    for (int i = 0; i < n; i++) {
        src[2*i] = (float)kB[m[i].b].x; src[2*i+1] = (float)kB[m[i].b].y;
        dst[2*i] = (float)kA[m[i].a].x; dst[2*i+1] = (float)kA[m[i].a].y;
    }
    std::vector<int> samples; make_ransac_samples(n, P::RANSAC_ITERS, samples);
    int bestCnt = -1; double bestH[9] = {0};
    for (int it = 0; it < P::RANSAC_ITERS; it++) {
        float s4[8], d4[8];
        for (int k = 0; k < 4; k++) {
            int id = samples[4*it + k];
            s4[2*k] = src[2*id]; s4[2*k+1] = src[2*id+1]; d4[2*k] = dst[2*id]; d4[2*k+1] = dst[2*id+1];
        }
        if (degenerate4(s4) || degenerate4(d4)) continue;
        double h[9];
        if (!estimate_H<double>(s4, d4, 4, h)) continue;
        int c = count_inliers(h, src.data(), dst.data(), n, thr);
        if (c > bestCnt) { bestCnt = c; for (int i = 0; i < 9; i++) bestH[i] = h[i]; }   // ties: lowest iteration wins
    }
    if (bestCnt < P::MIN_INLIERS) return false;
    for (int i = 0; i < 9; i++) H[i] = bestH[i];
    for (int r = 0; r < P::REFINE_ROUNDS; r++) {          // refit on all inliers
        std::vector<float> is, id;
        for (int i = 0; i < n; i++)
            if (count_inliers(H, &src[2*i], &dst[2*i], 1, thr)) {
                is.push_back(src[2*i]); is.push_back(src[2*i+1]); id.push_back(dst[2*i]); id.push_back(dst[2*i+1]); }
        double h[9];
        if (is.size() >= 8 && estimate_H<double>(is.data(), id.data(), (int)is.size() / 2, h))
            for (int i = 0; i < 9; i++) H[i] = h[i];
    }
    nin = count_inliers(H, src.data(), dst.data(), n, thr);
    return nin >= P::MIN_INLIERS;
}

static bool invert3x3(const double m[9], double inv[9]) {
    double c0 = m[4]*m[8]-m[5]*m[7], c1 = m[5]*m[6]-m[3]*m[8], c2 = m[3]*m[7]-m[4]*m[6];
    double det = m[0]*c0 + m[1]*c1 + m[2]*c2;
    if (fabs(det) < 1e-18) return false;
    double id = 1.0 / det;
    inv[0] = c0*id;                   inv[1] = (m[2]*m[7]-m[1]*m[8])*id; inv[2] = (m[1]*m[5]-m[2]*m[4])*id;
    inv[3] = c1*id;                   inv[4] = (m[0]*m[8]-m[2]*m[6])*id; inv[5] = (m[2]*m[3]-m[0]*m[5])*id;
    inv[6] = c2*id;                   inv[7] = (m[1]*m[6]-m[0]*m[7])*id; inv[8] = (m[0]*m[4]-m[1]*m[3])*id;
    return true;
}

// ---- Stage 0a: box downscale by an integer factor (used only for the focal-length search)
static Image downscale(const Image& in, int k) {
    Image o; o.w = in.w / k; o.h = in.h / k; o.rgb.resize((size_t)o.w * o.h * 3);
    for (int y = 0; y < o.h; y++)
        for (int x = 0; x < o.w; x++)
            for (int c = 0; c < 3; c++) {
                int s = 0;
                for (int dy = 0; dy < k; dy++) for (int dx = 0; dx < k; dx++) s += in.rgb[((size_t)(y*k+dy)*in.w + x*k + dx)*3 + c];
                o.rgb[((size_t)y*o.w + x)*3 + c] = (uint8_t)((s + k*k/2) / (k*k));
            }
    return o;
}

// ---- Stage 0b: cylindrical projection with focal length f (pixels), bilinear.
// Output column x' <-> angle t = (x'-cx')/f ; source = (f*tan t + cx, cy + (y'-cy')/cos t).
// Height is cropped to h*cos(theta_max) so that every output pixel has a valid source (no black wedges).
// A camera that only pans becomes a pure horizontal shift in this projection, so the homography stays well behaved.
static void cyl_project(const Image& in, double f, Image& out) {
    const double cx = (in.w - 1) * 0.5, cy = (in.h - 1) * 0.5;
    const double th = std::atan(in.w / (2.0 * f));
    const int W = (int)std::lround(2.0 * f * th), Hh = (int)std::floor(in.h * std::cos(th));
    out.w = W; out.h = Hh; out.rgb.assign((size_t)W * Hh * 3, 0);
    const double ox = (W - 1) * 0.5, oy = (Hh - 1) * 0.5;
    std::vector<int> cx0(W); std::vector<float> cfx(W); std::vector<double> cic(W);
    for (int x = 0; x < W; x++) {
        double t = (x - ox) / f, sx = f * std::tan(t) + cx;
        sx = std::min(std::max(sx, 0.0), (double)(in.w - 1));
        cx0[x] = std::min((int)sx, in.w - 2); cfx[x] = (float)(sx - cx0[x]); cic[x] = 1.0 / std::cos(t);
    }
    for (int y = 0; y < Hh; y++)
        for (int x = 0; x < W; x++) {
            double sy = cy + (y - oy) * cic[x];
            sy = std::min(std::max(sy, 0.0), (double)(in.h - 1));
            int y0 = std::min((int)sy, in.h - 2); float fy = (float)(sy - y0), fx = cfx[x];
            const uint8_t* p = &in.rgb[((size_t)y0*in.w + cx0[x]) * 3];
            for (int c = 0; c < 3; c++) {
                float top = p[c]*(1-fx) + p[3+c]*fx, bot = p[in.w*3+c]*(1-fx) + p[in.w*3+3+c]*fx;
                out.rgb[((size_t)y*W + x)*3 + c] = (uint8_t)(top*(1-fy) + bot*fy + 0.5f);
            }
        }
}

// ---- Stage 6a: seam finding. Returns tB[i] = weight of image B (0..1) for every canvas pixel in the overlap.
// Idea: inside the overlap, find the minimum-cost path (dynamic programming) where cost = how much A and B
// disagree (colour + gradient). The path avoids objects that parallax displaced between the two views, so
// each object is taken from ONE image only -> no double image / blur. A narrow cross-fade hides the seam.
static void seam_weights(int W, int H, const std::vector<uint8_t>& vA, const std::vector<uint8_t>& vB,
                         const std::vector<float>& cA, const std::vector<float>& cB, std::vector<float>& tB, float feather) {
    const size_t N = (size_t)W * H;
    tB.assign(N, 0.f);
    int x0 = W, x1 = -1, y0 = H, y1 = -1;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            if (vA[(size_t)y*W + x] && vB[(size_t)y*W + x]) {
                x0 = std::min(x0, x); x1 = std::max(x1, x); y0 = std::min(y0, y); y1 = std::max(y1, y); }
    if (x1 < 0) return;                                         // no overlap
    auto ov = [&](int x, int y) { return x >= 0 && x < W && y >= 0 && y < H && vA[(size_t)y*W + x] && vB[(size_t)y*W + x]; };
    auto lum = [&](const std::vector<float>& c, int x, int y) {
        size_t i = ((size_t)y*W + x) * 3; return 0.299f*c[i] + 0.587f*c[i+1] + 0.114f*c[i+2]; };

    // seam direction: tall overlap band (A left / B right) -> vertical seam (one x per row); wide band -> horizontal seam
    const bool vert = (x1 - x0 + 1) <= (y1 - y0 + 1);
    const int nMaj = vert ? H : W, nMin = vert ? W : H;
    auto pix = [&](int maj, int mn) { return vert ? (size_t)maj*W + mn : (size_t)mn*W + maj; };
    const int m0 = vert ? y0 : x0, m1 = vert ? y1 : x1, n0 = vert ? x0 : y0, n1 = vert ? x1 : y1;

    // which image lies on the high-coordinate side of the seam? (compare centroids along the minor axis)
    double sa = 0, sb = 0; size_t na = 0, nb = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            size_t i = (size_t)y*W + x; double m = vert ? x : y;
            if (vA[i]) { sa += m; na++; }
            if (vB[i]) { sb += m; nb++; }
        }
    const bool bHigh = (sb / std::max<size_t>(nb, 1)) > (sa / std::max<size_t>(na, 1));

    // overlap interval [lo,hi] of every major line (convex overlap => one interval)
    std::vector<int> lo(nMaj, INT_MAX), hi(nMaj, -1);
    for (int maj = m0; maj <= m1; maj++)
        for (int mn = n0; mn <= n1; mn++) {
            int x = vert ? mn : maj, y = vert ? maj : mn;
            if (ov(x, y)) { lo[maj] = std::min(lo[maj], mn); hi[maj] = std::max(hi[maj], mn); }
        }

    // cost map: |colour diff| + w * |gradient diff| + small centre bias
    std::vector<float> E(N, 0.f);
    for (int maj = m0; maj <= m1; maj++) {
        if (hi[maj] < 0) continue;
        float mid = 0.5f * (lo[maj] + hi[maj]), half = 0.5f * (hi[maj] - lo[maj]) + 1.f;
        for (int mn = lo[maj]; mn <= hi[maj]; mn++) {
            int x = vert ? mn : maj, y = vert ? maj : mn; size_t i = (size_t)y*W + x;
            float d = (fabsf(cA[3*i]-cB[3*i]) + fabsf(cA[3*i+1]-cB[3*i+1]) + fabsf(cA[3*i+2]-cB[3*i+2])) / 3.f;
            float g = 0.f;
            if (ov(x-1, y) && ov(x+1, y) && ov(x, y-1) && ov(x, y+1)) {
                float gxA = lum(cA, x+1, y) - lum(cA, x-1, y), gyA = lum(cA, x, y+1) - lum(cA, x, y-1);
                float gxB = lum(cB, x+1, y) - lum(cB, x-1, y), gyB = lum(cB, x, y+1) - lum(cB, x, y-1);
                g = fabsf(gxA - gxB) + fabsf(gyA - gyB);
            }
            E[i] = d + P::SEAM_GRAD_W * g + P::SEAM_CENTER_BIAS * fabsf(mn - mid) / half;
        }
    }

    // dynamic programming along the major axis (moves of -1/0/+1 on the minor axis)
    std::vector<float> D(N, 0.f); std::vector<int> back(N, 0), rows;
    for (int mn = lo[m0]; mn <= hi[m0]; mn++) D[pix(m0, mn)] = E[pix(m0, mn)];
    rows.push_back(m0); int pm = m0;
    for (int maj = m0 + 1; maj <= m1; maj++) {
        if (hi[maj] < 0) continue;
        for (int mn = lo[maj]; mn <= hi[maj]; mn++) {
            float best = 1e30f; int bp = lo[pm];
            for (int dl = -1; dl <= 1; dl++) {
                int p = clampi(mn + dl, lo[pm], hi[pm]);
                float c = D[pix(pm, p)] + (dl ? P::SEAM_TURN_PEN : 0.f);
                if (c < best) { best = c; bp = p; }
            }
            D[pix(maj, mn)] = E[pix(maj, mn)] + best; back[pix(maj, mn)] = bp;
        }
        rows.push_back(maj); pm = maj;
    }
    std::vector<int> seam(nMaj, -1);
    for (int maj = m0; maj <= m1; maj++) if (hi[maj] >= 0) seam[maj] = (lo[maj] + hi[maj]) / 2;   // fallback
    int cur = lo[pm]; float bestEnd = 1e30f;
    for (int mn = lo[pm]; mn <= hi[pm]; mn++) if (D[pix(pm, mn)] < bestEnd) { bestEnd = D[pix(pm, mn)]; cur = mn; }
    for (int k = (int)rows.size() - 1; k >= 0; k--) { seam[rows[k]] = cur; if (k > 0) cur = back[pix(rows[k], cur)]; }

    // weights: hard side choice + narrow linear cross-fade of width SEAM_FEATHER centred on the seam
    const float F = std::max(feather, 1.f);
    for (int maj = m0; maj <= m1; maj++) {
        if (hi[maj] < 0) continue;
        for (int mn = lo[maj]; mn <= hi[maj]; mn++) {
            float t = ((float)(mn - seam[maj]) + 0.5f) / F + 0.5f;        // weight of the high-coordinate side
            t = t < 0.f ? 0.f : (t > 1.f ? 1.f : t);
            tB[pix(maj, mn)] = bHigh ? t : 1.f - t;
        }
    }
    (void)nMin;
}

// ---- Stage 6b: multiband (Laplacian pyramid) blending.
// Low frequencies are mixed over a wide band around the seam, fine detail only over a narrow one, so smooth
// brightness / colour differences (sky gradient, vignetting) vanish without ghosting the structures.
struct Plane { int w = 0, h = 0, c = 1; std::vector<float> v; };

// 5-tap binomial blur + decimation by 2 (clamp-to-edge); output size (w+1)/2 x (h+1)/2
static Plane pyr_down(const Plane& s) {
    Plane d; d.c = s.c; d.w = (s.w + 1) / 2; d.h = (s.h + 1) / 2;
    Plane t; t.c = s.c; t.w = d.w; t.h = s.h; t.v.resize((size_t)t.w * t.h * t.c);
    for (int y = 0; y < s.h; y++)
        for (int x = 0; x < t.w; x++)
            for (int ch = 0; ch < s.c; ch++) {
                float a = 0;
                for (int k = -2; k <= 2; k++) a += P::BLUR_W[k+2] * s.v[((size_t)y*s.w + clampi(2*x + k, 0, s.w - 1)) * s.c + ch];
                t.v[((size_t)y*t.w + x) * t.c + ch] = a * (1.f / 16.f);
            }
    d.v.resize((size_t)d.w * d.h * d.c);
    for (int y = 0; y < d.h; y++)
        for (int x = 0; x < d.w; x++)
            for (int ch = 0; ch < s.c; ch++) {
                float a = 0;
                for (int k = -2; k <= 2; k++) a += P::BLUR_W[k+2] * t.v[((size_t)clampi(2*y + k, 0, t.h - 1)*t.w + x) * t.c + ch];
                d.v[((size_t)y*d.w + x) * d.c + ch] = a * (1.f / 16.f);
            }
    return d;
}

// bilinear upsampling to an exact target size
static Plane pyr_up(const Plane& s, int w, int h) {
    Plane u; u.c = s.c; u.w = w; u.h = h; u.v.resize((size_t)w * h * s.c);
    for (int y = 0; y < h; y++) {
        float fy = std::min(std::max((y + 0.5f) * 0.5f - 0.5f, 0.f), (float)(s.h - 1));
        int y0 = std::min((int)fy, std::max(s.h - 2, 0)), y1 = std::min(y0 + 1, s.h - 1); float wy = fy - y0;
        for (int x = 0; x < w; x++) {
            float fx = std::min(std::max((x + 0.5f) * 0.5f - 0.5f, 0.f), (float)(s.w - 1));
            int x0 = std::min((int)fx, std::max(s.w - 2, 0)), x1 = std::min(x0 + 1, s.w - 1); float wx = fx - x0;
            for (int ch = 0; ch < s.c; ch++) {
                float a = s.v[((size_t)y0*s.w + x0)*s.c + ch]*(1-wx) + s.v[((size_t)y0*s.w + x1)*s.c + ch]*wx;
                float b = s.v[((size_t)y1*s.w + x0)*s.c + ch]*(1-wx) + s.v[((size_t)y1*s.w + x1)*s.c + ch]*wx;
                u.v[((size_t)y*w + x)*s.c + ch] = a*(1-wy) + b*wy;
            }
        }
    }
    return u;
}

// out = A*(1-mask) + B*mask  (mask in [0,1], 1 channel) with pyramid blending; A, B are 3-channel planes of equal size
static Plane multiband_blend(Plane A, Plane B, Plane M, int levels) {
    std::vector<Plane> GA(1), GB(1), GM(1);
    GA[0] = std::move(A); GB[0] = std::move(B); GM[0] = std::move(M);
    for (int k = 0; k < levels; k++) { GA.push_back(pyr_down(GA[k])); GB.push_back(pyr_down(GB[k])); GM.push_back(pyr_down(GM[k])); }
    std::vector<Plane> BL(levels + 1);
    {   Plane& t = BL[levels]; t.w = GA[levels].w; t.h = GA[levels].h; t.c = 3; t.v.resize(GA[levels].v.size());
        for (size_t i = 0; i < (size_t)t.w * t.h; i++) { float m = GM[levels].v[i];
            for (int c = 0; c < 3; c++) t.v[3*i + c] = (1.f - m) * GA[levels].v[3*i + c] + m * GB[levels].v[3*i + c]; } }
    for (int k = levels - 1; k >= 0; k--) {
        Plane ua = pyr_up(GA[k+1], GA[k].w, GA[k].h), ub = pyr_up(GB[k+1], GB[k].w, GB[k].h);
        Plane& t = BL[k]; t.w = GA[k].w; t.h = GA[k].h; t.c = 3; t.v.resize(GA[k].v.size());
        for (size_t i = 0; i < (size_t)t.w * t.h; i++) { float m = GM[k].v[i];
            for (int c = 0; c < 3; c++)
                t.v[3*i + c] = (1.f - m) * (GA[k].v[3*i + c] - ua.v[3*i + c]) + m * (GB[k].v[3*i + c] - ub.v[3*i + c]); }
        GA[k + 1] = Plane(); GB[k + 1] = Plane(); GM[k] = Plane();               // free memory early (level k+1 is no longer needed)
    }
    Plane res = std::move(BL[levels]);
    for (int k = levels - 1; k >= 0; k--) {
        Plane u = pyr_up(res, BL[k].w, BL[k].h);
        for (size_t i = 0; i < u.v.size(); i++) u.v[i] += BL[k].v[i];
        res = std::move(u);
    }
    return res;
}

// fill invalid pixels with the nearest valid colour (same row, else nearest filled row) so the pyramid does not blur black into the image
static void fill_invalid(std::vector<float>& img, int W, int H, const std::vector<uint8_t>& valid) {
    std::vector<uint8_t> rowOk(H, 0);
    for (int y = 0; y < H; y++) {
        int first = -1;
        for (int x = 0; x < W; x++) if (valid[(size_t)y*W + x]) { first = x; break; }
        if (first < 0) continue;
        rowOk[y] = 1; int lastv = first;
        for (int x = 0; x < W; x++) {
            size_t i = (size_t)y*W + x;
            if (valid[i]) { lastv = x; continue; }
            int src = x < first ? first : lastv;
            for (int c = 0; c < 3; c++) img[3*i + c] = img[3*((size_t)y*W + src) + c];
        }
    }
    for (int y = 0; y < H; y++) {
        if (rowOk[y]) continue;
        int from = -1;
        for (int d = 1; d < H && from < 0; d++) { if (y - d >= 0 && rowOk[y-d]) from = y - d; else if (y + d < H && rowOk[y+d]) from = y + d; }
        if (from >= 0) for (int x = 0; x < W * 3; x++) img[(size_t)y*W*3 + x] = img[(size_t)from*W*3 + x];
    }
}

// ---- Stage 6b: canvas from H, inverse warp of B (bilinear), then blend (seam-cut or legacy feather)
static bool warp_blend(const Image& A, const Image& B, const double H[9], Pano& out) {
    double cx[4] = {0, (double)B.w-1, (double)B.w-1, 0}, cy[4] = {0, 0, (double)B.h-1, (double)B.h-1};
    double minx = 0, miny = 0, maxx = A.w-1, maxy = A.h-1;
    for (int i = 0; i < 4; i++) {
        double w = H[6]*cx[i] + H[7]*cy[i] + H[8];
        if (w <= 1e-9) return false;                      // corner behind the camera plane
        double X = (H[0]*cx[i] + H[1]*cy[i] + H[2]) / w, Y = (H[3]*cx[i] + H[4]*cy[i] + H[5]) / w;
        minx = std::min(minx, X); maxx = std::max(maxx, X); miny = std::min(miny, Y); maxy = std::max(maxy, Y);
    }
    int ox = (int)floor(minx), oy = (int)floor(miny);
    int W = (int)ceil(maxx) - ox + 1, Hh = (int)ceil(maxy) - oy + 1;
    if (W > P::MAX_CANVAS_FACTOR * std::max(A.w, B.w) || Hh > P::MAX_CANVAS_FACTOR * std::max(A.h, B.h)) return false;
    double Hi[9]; if (!invert3x3(H, Hi)) return false;
    out.w = W; out.h = Hh; out.rgb.assign((size_t)W * Hh * 3, 0);

    // pass 1: both images resampled onto the canvas (validity mask, colour, border-distance weight)
    const size_t N = (size_t)W * Hh;
    std::vector<uint8_t> vA(N, 0), vB(N, 0);
    std::vector<float> cA(N * 3, 0.f), cB(N * 3, 0.f), wA(N, 0.f), wB(N, 0.f);
    for (int y = 0; y < Hh; y++)
        for (int x = 0; x < W; x++) {
            int X = x + ox, Y = y + oy; size_t i = (size_t)y*W + x;
            if (X >= 0 && X < A.w && Y >= 0 && Y < A.h) {
                vA[i] = 1;
                wA[i] = (float)(std::min(std::min(X, A.w-1-X), std::min(Y, A.h-1-Y)) + 1);
                for (int c = 0; c < 3; c++) cA[3*i + c] = A.rgb[((size_t)Y*A.w + X)*3 + c];
            }
            double pw = Hi[6]*X + Hi[7]*Y + Hi[8];
            if (fabs(pw) > 1e-12) {
                double u = (Hi[0]*X + Hi[1]*Y + Hi[2]) / pw, v = (Hi[3]*X + Hi[4]*Y + Hi[5]) / pw;
                if (u >= 0 && u <= B.w-1 && v >= 0 && v <= B.h-1) {
                    int u0 = std::min((int)u, B.w-2), v0 = std::min((int)v, B.h-2);
                    float fu = (float)(u - u0), fv = (float)(v - v0);
                    vB[i] = 1;
                    wB[i] = (float)(std::min(std::min(u, B.w-1-u), std::min(v, B.h-1-v)) + 1);
                    for (int c = 0; c < 3; c++) {
                        const uint8_t* p = &B.rgb[((size_t)v0*B.w + u0)*3 + c];
                        float top = p[0]*(1-fu) + p[3]*fu, bot = p[B.w*3]*(1-fu) + p[B.w*3+3]*fu;
                        cB[3*i + c] = top*(1-fv) + bot*fv;
                    }
                }
            }
        }

    // pass 1b: exposure compensation - per-channel gain on B so that the overlap means agree (saturated/black pixels ignored)
    if (P::EXPOSURE_COMP) {
        double sA[3] = {0,0,0}, sB[3] = {0,0,0}; size_t cnt = 0;
        for (size_t i = 0; i < N; i++) {
            if (!(vA[i] && vB[i])) continue;
            float la = 0.299f*cA[3*i] + 0.587f*cA[3*i+1] + 0.114f*cA[3*i+2], lb = 0.299f*cB[3*i] + 0.587f*cB[3*i+1] + 0.114f*cB[3*i+2];
            if (la < 15 || la > 240 || lb < 15 || lb > 240) continue;
            for (int c = 0; c < 3; c++) { sA[c] += cA[3*i+c]; sB[c] += cB[3*i+c]; }
            cnt++;
        }
        if (cnt > 1000) {
            for (int c = 0; c < 3; c++) {
                float g = (float)(sA[c] / std::max(sB[c], 1.0)); g = g < P::GAIN_MIN ? P::GAIN_MIN : (g > P::GAIN_MAX ? P::GAIN_MAX : g);
                for (size_t i = 0; i < N; i++) if (vB[i]) cB[3*i+c] *= g;
            }
        }
    }

    // pass 2: blend weight of B per pixel
    std::vector<float> tB;
    if (P::SEAM_BLEND) seam_weights(W, Hh, vA, vB, cA, cB, tB, P::MULTIBAND ? 1.f : (float)P::SEAM_FEATHER);
    if (P::SEAM_BLEND && P::MULTIBAND) {
        std::vector<uint8_t> vU(N);
        for (size_t i = 0; i < N; i++) vU[i] = vA[i] | vB[i];
        // extend each image beyond its own border with its own edge colour, so the pyramids contain no artificial step there
        fill_invalid(cA, W, Hh, vA); fill_invalid(cB, W, Hh, vB);
        Plane pa, pb, pm; pa.w = pb.w = pm.w = W; pa.h = pb.h = pm.h = Hh; pa.c = pb.c = 3; pm.c = 1;
        pa.v = std::move(cA); pb.v = std::move(cB); pm.v.resize(N);
        for (size_t i = 0; i < N; i++) pm.v[i] = !vB[i] ? 0.f : (!vA[i] ? 1.f : tB[i]);
        int lv = P::MB_LEVELS; while (lv > 1 && (W >> lv) < 8) lv--; while (lv > 1 && (Hh >> lv) < 8) lv--;
        Plane res = multiband_blend(std::move(pa), std::move(pb), std::move(pm), lv);
        for (size_t i = 0; i < N; i++) {
            if (!vU[i]) continue;
            for (int c = 0; c < 3; c++) { float px = res.v[3*i + c] + 0.5f; out.rgb[3*i + c] = (uint8_t)(px < 0.f ? 0.f : (px > 255.f ? 255.f : px)); }
        }
        return true;
    }
    for (size_t i = 0; i < N; i++) {
        if (!vA[i] && !vB[i]) continue;
        float t;
        if (!vB[i]) t = 0.f;
        else if (!vA[i]) t = 1.f;
        else t = P::SEAM_BLEND ? tB[i] : wB[i] / (wA[i] + wB[i]);       // overlap
        for (int c = 0; c < 3; c++)
        { float px = (1.f - t) * cA[3*i + c] + t * cB[3*i + c] + 0.5f; out.rgb[3*i + c] = (uint8_t)(px < 0.f ? 0.f : (px > 255.f ? 255.f : px)); }
    }
    return true;
}

// ---- Stages 1-5 on an image pair (gray -> blur -> FAST -> BRIEF -> match -> RANSAC). t == nullptr: no timing.
static bool register_images(const Image& A, const Image& B, StageTimes* t, Stats& s, double thr) {
    std::vector<uint8_t> gA, gB, bA, bB;
    std::vector<Keypoint> kA, kB; std::vector<Descriptor> dA, dB; std::vector<Match> m;
    auto t0 = Clock::now();
    auto lap = [&](double StageTimes::*field) { if (t) t->*field = ms_since(t0); t0 = Clock::now(); };
    to_gray(A, gA); to_gray(B, gB);                              lap(&StageTimes::gray);
    gauss_blur(gA, A.w, A.h, bA); gauss_blur(gB, B.w, B.h, bB);  lap(&StageTimes::blur);
    detect_fast(bA, A.w, A.h, kA); detect_fast(bB, B.w, B.h, kB); lap(&StageTimes::fast);
    brief_describe(bA, A.w, kA, dA); brief_describe(bB, B.w, kB, dB); lap(&StageTimes::brief);
    match_bf(dA, dB, m);                                         lap(&StageTimes::match);
    s.kpA = (int)kA.size(); s.kpB = (int)kB.size(); s.matches = (int)m.size();
    bool ok = ransac_homography(kA, kB, m, s.H, s.inliers, thr); lap(&StageTimes::ransac);
    return ok;
}

// Footprint distortion of H on a w x h image: ratio between the largest and smallest local area magnification
// (|det H| / w^3) over the corners and centre. ~1 for a (near) translation, huge for a planar fit after a big rotation.
static double footprint_distortion(const double H[9], int w, int h) {
    double det = H[0]*(H[4]*H[8]-H[5]*H[7]) - H[1]*(H[3]*H[8]-H[5]*H[6]) + H[2]*(H[3]*H[7]-H[4]*H[6]);
    const double px[5] = {0, (double)w-1, (double)w-1, 0, (w-1)*0.5}, py[5] = {0, 0, (double)h-1, (double)h-1, (h-1)*0.5};
    double mn = 1e300, mx = 0;
    for (int i = 0; i < 5; i++) {
        double ww = H[6]*px[i] + H[7]*py[i] + H[8];
        if (ww <= 1e-9) return 1e9;
        double m = fabs(det) / (ww*ww*ww); mn = std::min(mn, m); mx = std::max(mx, m);
    }
    return mn > 0 ? mx / mn : 1e9;
}

// ---- Focal-length search on downscaled copies: returns the best focal/width fraction, 0 = planar.
// Raw inlier counts are biased (a cylinder image is cropped, planar is not), so they only act as a filter
// (>= FOCAL_KEEP * best); the choice itself minimises the footprint distortion of the estimated homography.
static double choose_focal(const Image& A, const Image& B) {
    Image a = downscale(A, P::FOCAL_DOWNSCALE), b = downscale(B, P::FOCAL_DOWNSCALE);
    double frac[P::FOCAL_N + 1], dist[P::FOCAL_N + 1]; int inl[P::FOCAL_N + 1]; int maxInl = 0;
    for (int i = 0; i <= P::FOCAL_N; i++) {                       // last candidate = planar
        frac[i] = i < P::FOCAL_N ? P::FOCAL_FRAC[i] : 0.0; inl[i] = 0; dist[i] = 1e18;
        Image ca, cb; const Image *pa = &a, *pb = &b;
        if (frac[i] > 0) { cyl_project(a, frac[i] * a.w, ca); cyl_project(b, frac[i] * b.w, cb); pa = &ca; pb = &cb; }
        Stats st;
        if (register_images(*pa, *pb, nullptr, st, P::FOCAL_THRESH)) {
            inl[i] = st.inliers; dist[i] = footprint_distortion(st.H, pa->w, pa->h); maxInl = std::max(maxInl, inl[i]);
        }
    }
    double bestFrac = 0.0, bestD = 1e18;                          // nothing worked => planar
    for (int i = 0; i <= P::FOCAL_N; i++)
        if (inl[i] > 0 && inl[i] >= P::FOCAL_KEEP * maxInl && dist[i] < bestD) { bestD = dist[i]; bestFrac = frac[i]; }
    return bestFrac;
}

// ---- Driver with per-stage timing (std::chrono). Stages 1-3 are summed over both images.
bool run_v0(const Image& A0, const Image& B0, Pano& out, StageTimes& t, Stats& s) {
    t = StageTimes(); s = Stats();
    auto T0 = Clock::now();
    auto t0 = Clock::now();
    const double frac = g_focal_frac < 0 ? choose_focal(A0, B0) : g_focal_frac;
    Image Ac, Bc; const Image *A = &A0, *B = &B0;
    if (frac > 0) { cyl_project(A0, frac * A0.w, Ac); cyl_project(B0, frac * B0.w, Bc); A = &Ac; B = &Bc; }
    t.proj = ms_since(t0);
    s.focal_frac = frac;
    bool ok = register_images(*A, *B, &t, s, P::RANSAC_THRESH);
    t0 = Clock::now();
    if (ok) ok = warp_blend(*A, *B, s.H, out);
    t.warp = ms_since(t0);
    t.total = ms_since(T0);
    s.ok = ok;
    return ok;
}
