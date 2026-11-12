// common.h - parameters, data structures and math shared by ALL versions (V0, V1, ...)
// Everything that affects results lives here so every version uses identical settings.
// Authors: <TODO: Ten - MSSV, ...>
// External sources: none (<TODO: sua neu co doan code lay/chinh tu nguon ngoai>)
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
constexpr int BORDER   = 22;     // keypoints must be >= BORDER px from the image edge (rotated BRIEF pattern reaches 21 px)
constexpr int TOPK = 8000;   // keypoints kept per image
// Stage 3: BRIEF-256 on the blurred image
constexpr int BRIEF_BITS  = 256;
constexpr int BRIEF_WORDS = 8;   // 8 x uint32
constexpr uint32_t BRIEF_SEED = 12345u;
// Oriented BRIEF (ORB-style): the keypoint orientation is the direction of the intensity centroid of a disc of radius
// ORI_RADIUS, quantised to ORI_BINS directions; the 256-pair pattern is pre-rotated once per bin (integer tables).
constexpr int ORI_BINS   = 30;   // 12 degree steps
constexpr int ORI_RADIUS = 15;
// Stage 4: matching (Hamming, ratio test, cross-check)
constexpr int MAX_HAMMING = 80;
constexpr int RATIO_NUM = 4, RATIO_DEN = 5;   // best/second < 0.8, integer arithmetic
// Stage 5: RANSAC
constexpr int   RANSAC_ITERS  = 2000;
constexpr uint32_t RANSAC_SEED = 2024u;
constexpr double RANSAC_THRESH = 3.0;         // pixels, reprojection error in image A
constexpr int   MIN_INLIERS   = 12;
constexpr int   REFINE_ROUNDS = 2;
// Stage 0: cylindrical projection (undoes the huge perspective stretch when the camera rotates a lot between shots).
// The focal length is unknown (no EXIF), so it is searched on 1/FOCAL_DOWNSCALE-size copies: for each candidate
// f = frac * image_width (plus "planar" = no projection) the whole registration is run. Among the candidates that keep
// at least FOCAL_KEEP of the best inlier count, the one whose homography is closest to a pure translation (smallest
// footprint distortion, see footprint_distortion()) wins. Override from the CLI with --focal <frac> (0 = planar).
constexpr int    FOCAL_DOWNSCALE = 4;
constexpr int    FOCAL_N = 9;
constexpr double FOCAL_FRAC[FOCAL_N] = {0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5};
constexpr double FOCAL_KEEP = 0.7;            // candidates must reach this fraction of the best inlier count
constexpr double FOCAL_THRESH = 2.0;          // RANSAC threshold (px) used on the downscaled copies during the search
// Stage 6: warp/blend
constexpr int MAX_CANVAS_FACTOR = 3;          // reject canvases larger than 3x the input size
constexpr bool  EXPOSURE_COMP    = true;      // scale B per channel so its mean matches A in the overlap (hides brightness steps)
constexpr float GAIN_MIN = 0.7f, GAIN_MAX = 1.4f;   // clamp of that per-channel gain
constexpr bool  MULTIBAND        = true;      // Laplacian-pyramid blending across the seam (hides smooth brightness/colour differences)
constexpr int   MB_LEVELS = 7;         // pyramid levels (capped by the canvas size)
constexpr bool  SEAM_BLEND       = true;      // true: seam-cut blend (no ghosting under parallax); false: old feather average
constexpr int   SEAM_FEATHER     = 12;        // px: width of the narrow cross-fade around the seam (1 = hard cut)
constexpr float SEAM_GRAD_W      = 0.5f;      // weight of the gradient-difference term in the seam cost
constexpr float SEAM_TURN_PEN    = 2.0f;      // penalty for the seam moving sideways (keeps it straighter)
constexpr float SEAM_CENTER_BIAS = 4.0f;      // mild preference for the middle of the overlap
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
    double proj = 0;               // focal search + cylindrical projection (appended last so the field order above is unchanged)
    double kernels() const { return proj + gray + blur + fast + brief + match + ransac + warp; }
};

struct Stats { int kpA = 0, kpB = 0, matches = 0, inliers = 0; double H[9] = {1,0,0,0,1,0,0,0,1}; bool ok = false;
               double focal_frac = 0; };   // focal length used / image width (0 = planar)

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

// ------------------------------------------------------------------ oriented BRIEF helpers
// direction table: (cos,sin) * 1024 of every orientation bin (host-built, shared with the GPU versions)
inline void make_ori_dirs(int dir[P::ORI_BINS][2]) {
    for (int b = 0; b < P::ORI_BINS; b++) {
        double a = 2.0 * 3.14159265358979323846 * b / P::ORI_BINS;
        dir[b][0] = (int)std::lround(std::cos(a) * 1024.0); dir[b][1] = (int)std::lround(std::sin(a) * 1024.0);
    }
}

// pre-rotated sampling patterns: out[(bin*BRIEF_BITS + i)*4 + k], k = x1,y1,x2,y2 (integer offsets, |v| <= 21)
inline void make_brief_patterns(std::vector<int>& out) {
    int base[P::BRIEF_BITS][4]; make_brief_pattern(base);
    out.assign((size_t)P::ORI_BINS * P::BRIEF_BITS * 4, 0);
    for (int b = 0; b < P::ORI_BINS; b++) {
        double a = 2.0 * 3.14159265358979323846 * b / P::ORI_BINS, c = std::cos(a), s = std::sin(a);
        for (int i = 0; i < P::BRIEF_BITS; i++)
            for (int k = 0; k < 4; k += 2) {
                double x = base[i][k], y = base[i][k+1];
                out[((size_t)b * P::BRIEF_BITS + i) * 4 + k]     = (int)std::lround(x * c - y * s);
                out[((size_t)b * P::BRIEF_BITS + i) * 4 + k + 1] = (int)std::lround(x * s + y * c);
            }
    }
}

// orientation bin of the intensity centroid (integer-only given the direction table => identical on every platform)
HD inline int keypoint_orientation(const uint8_t* img, int w, int x, int y, const int dir[P::ORI_BINS][2]) {
    long long m10 = 0, m01 = 0;
    for (int dy = -P::ORI_RADIUS; dy <= P::ORI_RADIUS; dy++)
        for (int dx = -P::ORI_RADIUS; dx <= P::ORI_RADIUS; dx++) {
            if (dx*dx + dy*dy > P::ORI_RADIUS * P::ORI_RADIUS) continue;
            int v = img[(size_t)(y + dy) * w + x + dx];
            m10 += (long long)dx * v; m01 += (long long)dy * v;
        }
    int best = 0; long long bs = -(1LL << 62);
    for (int b = 0; b < P::ORI_BINS; b++) {
        long long sc = m10 * dir[b][0] + m01 * dir[b][1];
        if (sc > bs) { bs = sc; best = b; }
    }
    return best;
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
