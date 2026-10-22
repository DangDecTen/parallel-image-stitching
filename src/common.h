// common.h - parameters, data structures and math shared by ALL versions (V0, V1, ...)
// Everything that affects results lives here so every version uses identical settings.
#pragma once
#include <cstdint>
#include <cmath>
#include <vector>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ------------------------------------------------------------------ parameters
namespace P {
// Stage 1: grayscale (integer weights, exact on CPU and GPU) + 5-tap binomial blur (sum 16, 2D sum 256)
constexpr int BLUR_W[5] = {1, 4, 6, 4, 1};
// Stage 2: FAST-9 on the blurred image
constexpr int FAST_T   = 20;     // contrast threshold
constexpr int BORDER   = 16;     // keypoints must be >= BORDER px from the image edge (BRIEF patch radius is 15)
constexpr int TOPK     = 4000;   // keypoints kept per image
// Stage 3: BRIEF-256 on the blurred image
constexpr int BRIEF_BITS  = 256;
constexpr int BRIEF_WORDS = 8;   // 8 x uint32
constexpr uint32_t BRIEF_SEED = 12345u;
// Stage 4: matching (Hamming, ratio test, cross-check)
constexpr int MAX_HAMMING = 80;
constexpr int RATIO_NUM = 4, RATIO_DEN = 5;   // best/second < 0.8, integer arithmetic
// Stage 5: RANSAC
constexpr int   RANSAC_ITERS  = 2000;
constexpr uint32_t RANSAC_SEED = 2024u;
constexpr double RANSAC_THRESH = 3.0;         // pixels, reprojection error in image A
constexpr int   MIN_INLIERS   = 12;
constexpr int   REFINE_ROUNDS = 2;
// Stage 6: warp/blend
constexpr int MAX_CANVAS_FACTOR = 3;          // reject canvases larger than 3x the input size
}

// FAST circle of radius 3 (16 pixels), clockwise from the top
static const int FAST_RING[16][2] = {
    {0,-3},{1,-3},{2,-2},{3,-1},{3,0},{3,1},{2,2},{1,3},
    {0,3},{-1,3},{-2,2},{-3,1},{-3,0},{-3,-1},{-2,-2},{-1,-3}};

// ------------------------------------------------------------------ data structures
struct Image { int w = 0, h = 0; std::vector<uint8_t> rgb; };   // interleaved RGB
struct Pano  { int w = 0, h = 0; std::vector<uint8_t> rgb; };

struct Keypoint   { int x, y, score; };
struct Descriptor { uint32_t w[P::BRIEF_WORDS]; };
struct Match      { int a, b, dist; };                           // index in A, index in B, Hamming distance

struct StageTimes {                // milliseconds
    double gray = 0, blur = 0, fast = 0, brief = 0, match = 0, ransac = 0, warp = 0;
    double h2d = 0, d2h = 0;       // transfers (0 for V0)
    double total = 0;              // end-to-end, excludes disk I/O
    double kernels() const { return gray + blur + fast + brief + match + ransac + warp; }
};

struct Stats { int kpA = 0, kpB = 0, matches = 0, inliers = 0; double H[9] = {1,0,0,0,1,0,0,0,1}; bool ok = false; };

// ------------------------------------------------------------------ deterministic RNG (xorshift32)
HD inline uint32_t xorshift32(uint32_t& s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }

// BRIEF sampling pattern: 256 pairs (x1,y1,x2,y2), each coordinate ~ bell-shaped in [-15,15]
// (sum of three uniform draws in [-5,5]). Integer-only so CPU/GPU tables are identical.
inline void make_brief_pattern(int pat[P::BRIEF_BITS][4]) {
    uint32_t s = P::BRIEF_SEED;
    for (int i = 0; i < P::BRIEF_BITS; i++)
        for (int k = 0; k < 4; k++) {
            int v = 0;
            for (int j = 0; j < 3; j++) v += (int)(xorshift32(s) % 11u) - 5;
            pat[i][k] = v;
        }
}

// Pre-generated RANSAC samples: iters x 4 DISTINCT match indices (same table is uploaded to the GPU later)
inline void make_ransac_samples(int n, int iters, std::vector<int>& out) {
    out.assign((size_t)iters * 4, 0);
    uint32_t s = P::RANSAC_SEED;
    for (int it = 0; it < iters; it++) {
        int idx[4];
        for (int k = 0; k < 4; k++) {
            bool dup;
            do { idx[k] = (int)(xorshift32(s) % (uint32_t)n); dup = false;
                 for (int j = 0; j < k; j++) if (idx[j] == idx[k]) dup = true; } while (dup);
            out[(size_t)it * 4 + k] = idx[k];
        }
    }
}

// ------------------------------------------------------------------ homography math (host+device)
// Points p = (x0,y0,x1,y1,...). src = points in B, dst = points in A. H maps B -> A.

// reject nearly collinear 4-point samples (area of any triple < 2 px^2)
HD inline bool degenerate4(const float* p) {
    const int T[4][3] = {{0,1,2},{0,1,3},{0,2,3},{1,2,3}};
    for (int t = 0; t < 4; t++) {
        int a = T[t][0], b = T[t][1], c = T[t][2];
        float cr = (p[2*b]-p[2*a])*(p[2*c+1]-p[2*a+1]) - (p[2*b+1]-p[2*a+1])*(p[2*c]-p[2*a]);
        if (fabsf(cr) < 4.0f) return true;
    }
    return false;
}

// Normalized DLT with h33 = 1, least squares via normal equations (exact solve when n == 4).
template <typename T>
HD inline bool estimate_H(const float* src, const float* dst, int n, T H[9]) {
    T cs0 = 0, cs1 = 0, cd0 = 0, cd1 = 0;
    for (int i = 0; i < n; i++) { cs0 += src[2*i]; cs1 += src[2*i+1]; cd0 += dst[2*i]; cd1 += dst[2*i+1]; }
    cs0 /= n; cs1 /= n; cd0 /= n; cd1 /= n;
    T ms = 0, md = 0;
    for (int i = 0; i < n; i++) {
        ms += sqrt((src[2*i]-cs0)*(src[2*i]-cs0) + (src[2*i+1]-cs1)*(src[2*i+1]-cs1));
        md += sqrt((dst[2*i]-cd0)*(dst[2*i]-cd0) + (dst[2*i+1]-cd1)*(dst[2*i+1]-cd1));
    }
    ms /= n; md /= n;
    if (ms < (T)1e-9 || md < (T)1e-9) return false;
    T ss = (T)1.4142135623730951 / ms, sd = (T)1.4142135623730951 / md;

    T M[8][9];
    for (int a = 0; a < 8; a++) for (int b = 0; b < 9; b++) M[a][b] = 0;
    for (int i = 0; i < n; i++) {
        T x = (src[2*i]-cs0)*ss, y = (src[2*i+1]-cs1)*ss, u = (dst[2*i]-cd0)*sd, v = (dst[2*i+1]-cd1)*sd;
        T r1[8] = {x, y, 1, 0, 0, 0, -u*x, -u*y};
        T r2[8] = {0, 0, 0, x, y, 1, -v*x, -v*y};
        for (int a = 0; a < 8; a++) {
            for (int b = 0; b < 8; b++) M[a][b] += r1[a]*r1[b] + r2[a]*r2[b];
            M[a][8] += r1[a]*u + r2[a]*v;
        }
    }
    // Gaussian elimination with partial pivoting
    for (int c = 0; c < 8; c++) {
        int p = c; T best = fabs(M[c][c]);
        for (int r = c+1; r < 8; r++) if (fabs(M[r][c]) > best) { best = fabs(M[r][c]); p = r; }
        if (best < (T)1e-12) return false;
        if (p != c) for (int k = 0; k < 9; k++) { T t = M[c][k]; M[c][k] = M[p][k]; M[p][k] = t; }
        for (int r = c+1; r < 8; r++) {
            T f = M[r][c] / M[c][c];
            for (int k = c; k < 9; k++) M[r][k] -= f * M[c][k];
        }
    }
    T h[8];
    for (int r = 7; r >= 0; r--) {
        T s = M[r][8];
        for (int k = r+1; k < 8; k++) s -= M[r][k] * h[k];
        h[r] = s / M[r][r];
    }
    // H = Td^-1 * Hn * Ts
    T Hn[9] = {h[0],h[1],h[2], h[3],h[4],h[5], h[6],h[7],1};
    T Ts[9] = {ss,0,-ss*cs0, 0,ss,-ss*cs1, 0,0,1};
    T Td[9] = {1/sd,0,cd0, 0,1/sd,cd1, 0,0,1};
    T tmp[9];
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
        T s = 0; for (int k = 0; k < 3; k++) s += Hn[i*3+k]*Ts[k*3+j]; tmp[i*3+j] = s; }
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
        T s = 0; for (int k = 0; k < 3; k++) s += Td[i*3+k]*tmp[k*3+j]; H[i*3+j] = s; }
    if (fabs(H[8]) < (T)1e-12) return false;
    T inv = 1 / H[8];
    for (int i = 0; i < 9; i++) H[i] *= inv;
    return true;
}

// Number of matches whose reprojection error (B->A) is below thr
HD inline int count_inliers(const double H[9], const float* src, const float* dst, int n, double thr) {
    int cnt = 0; double t2 = thr * thr;
    for (int i = 0; i < n; i++) {
        double x = src[2*i], y = src[2*i+1];
        double w = H[6]*x + H[7]*y + H[8];
        if (fabs(w) < 1e-12) continue;
        double u = (H[0]*x + H[1]*y + H[2]) / w - dst[2*i];
        double v = (H[3]*x + H[4]*y + H[5]) / w - dst[2*i+1];
        if (u*u + v*v < t2) cnt++;
    }
    return cnt;
}
