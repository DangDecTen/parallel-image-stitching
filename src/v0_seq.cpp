// v0_seq.cpp - V0: single-thread sequential reference implementation.
// Pipeline: gray -> blur -> FAST-9 + NMS + top-K -> BRIEF-256 -> brute-force Hamming match
//           -> RANSAC (4-pt DLT) + inlier refinement -> inverse warp + feather blend
#include "stitch.h"
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

// ---- Stage 3: BRIEF-256 (bit = I(p1) < I(p2) on the blurred image)
static void brief_describe(const std::vector<uint8_t>& img, int w, const std::vector<Keypoint>& kps,
                           std::vector<Descriptor>& desc) {
    int pat[P::BRIEF_BITS][4]; make_brief_pattern(pat);
    desc.assign(kps.size(), Descriptor());
    for (size_t i = 0; i < kps.size(); i++) {
        int x = kps[i].x, y = kps[i].y;
        for (int b = 0; b < P::BRIEF_BITS; b++) {
            int v1 = img[(size_t)(y + pat[b][1])*w + x + pat[b][0]];
            int v2 = img[(size_t)(y + pat[b][3])*w + x + pat[b][2]];
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
                              const std::vector<Match>& m, double H[9], int& nin) {
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
        int c = count_inliers(h, src.data(), dst.data(), n, P::RANSAC_THRESH);
        if (c > bestCnt) { bestCnt = c; for (int i = 0; i < 9; i++) bestH[i] = h[i]; }   // ties: lowest iteration wins
    }
    if (bestCnt < P::MIN_INLIERS) return false;
    for (int i = 0; i < 9; i++) H[i] = bestH[i];
    for (int r = 0; r < P::REFINE_ROUNDS; r++) {          // refit on all inliers
        std::vector<float> is, id;
        for (int i = 0; i < n; i++)
            if (count_inliers(H, &src[2*i], &dst[2*i], 1, P::RANSAC_THRESH)) {
                is.push_back(src[2*i]); is.push_back(src[2*i+1]); id.push_back(dst[2*i]); id.push_back(dst[2*i+1]); }
        double h[9];
        if (is.size() >= 8 && estimate_H<double>(is.data(), id.data(), (int)is.size() / 2, h))
            for (int i = 0; i < 9; i++) H[i] = h[i];
    }
    nin = count_inliers(H, src.data(), dst.data(), n, P::RANSAC_THRESH);
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

// ---- Stage 6: canvas from H, inverse warp of B (bilinear), feathered blend in the overlap
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
    for (int y = 0; y < Hh; y++)
        for (int x = 0; x < W; x++) {
            int X = x + ox, Y = y + oy;
            float wa = 0, wb = 0, ca[3] = {0,0,0}, cb[3] = {0,0,0};
            if (X >= 0 && X < A.w && Y >= 0 && Y < A.h) {
                wa = (float)(std::min(std::min(X, A.w-1-X), std::min(Y, A.h-1-Y)) + 1);
                for (int c = 0; c < 3; c++) ca[c] = A.rgb[((size_t)Y*A.w + X)*3 + c];
            }
            double pw = Hi[6]*X + Hi[7]*Y + Hi[8];
            if (fabs(pw) > 1e-12) {
                double u = (Hi[0]*X + Hi[1]*Y + Hi[2]) / pw, v = (Hi[3]*X + Hi[4]*Y + Hi[5]) / pw;
                if (u >= 0 && u <= B.w-1 && v >= 0 && v <= B.h-1) {
                    int u0 = std::min((int)u, B.w-2), v0 = std::min((int)v, B.h-2);
                    float fu = (float)(u - u0), fv = (float)(v - v0);
                    wb = (float)(std::min(std::min(u, B.w-1-u), std::min(v, B.h-1-v)) + 1);
                    for (int c = 0; c < 3; c++) {
                        const uint8_t* p = &B.rgb[((size_t)v0*B.w + u0)*3 + c];
                        float top = p[0]*(1-fu) + p[3]*fu, bot = p[B.w*3]*(1-fu) + p[B.w*3+3]*fu;
                        cb[c] = top*(1-fv) + bot*fv;
                    }
                }
            }
            if (wa + wb <= 0) continue;
            for (int c = 0; c < 3; c++)
                out.rgb[((size_t)y*W + x)*3 + c] = (uint8_t)((wa*ca[c] + wb*cb[c]) / (wa + wb) + 0.5f);
        }
    return true;
}

// ---- Driver with per-stage timing (std::chrono). Stages 1-3 are summed over both images.
bool run_v0(const Image& A, const Image& B, Pano& out, StageTimes& t, Stats& s) {
    t = StageTimes(); s = Stats();
    auto T0 = Clock::now();
    std::vector<uint8_t> gA, gB, bA, bB;
    std::vector<Keypoint> kA, kB; std::vector<Descriptor> dA, dB; std::vector<Match> m;
    auto t0 = Clock::now();
    to_gray(A, gA); to_gray(B, gB);                        t.gray  = ms_since(t0); t0 = Clock::now();
    gauss_blur(gA, A.w, A.h, bA); gauss_blur(gB, B.w, B.h, bB); t.blur = ms_since(t0); t0 = Clock::now();
    detect_fast(bA, A.w, A.h, kA); detect_fast(bB, B.w, B.h, kB); t.fast = ms_since(t0); t0 = Clock::now();
    brief_describe(bA, A.w, kA, dA); brief_describe(bB, B.w, kB, dB); t.brief = ms_since(t0); t0 = Clock::now();
    match_bf(dA, dB, m);                                   t.match = ms_since(t0); t0 = Clock::now();
    s.kpA = (int)kA.size(); s.kpB = (int)kB.size(); s.matches = (int)m.size();
    bool ok = ransac_homography(kA, kB, m, s.H, s.inliers); t.ransac = ms_since(t0); t0 = Clock::now();
    if (ok) ok = warp_blend(A, B, s.H, out);
    t.warp = ms_since(t0);
    t.total = ms_since(T0);
    s.ok = ok;
    return ok;
}
