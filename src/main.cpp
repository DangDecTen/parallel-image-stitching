// main.cpp - CLI, benchmark loop, reporting.  ./stitch <left> <right> <out> --version <0|1|..> [--runs N]
#include "stitch.h"
#include "io.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <left> <right> <out> --version <0|1|...> [--runs N]\n", argv[0]); return 1; }
    int version = 0, runs = 5;
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--version") && i + 1 < argc) version = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--runs") && i + 1 < argc) runs = atoi(argv[++i]);
    }
    Image A, B;
    if (!load_image(argv[1], A) || !load_image(argv[2], B)) { fprintf(stderr, "cannot load input images\n"); return 1; }

    bool (*fn)(const Image&, const Image&, Pano&, StageTimes&, Stats&) = nullptr;
    switch (version) { case 0: fn = run_v0; break; default: fprintf(stderr, "version %d not implemented yet\n", version); return 1; }

    Pano pano; StageTimes t; Stats st;
    fn(A, B, pano, t, st);                                   // warm-up, not timed
    StageTimes sum, mn; bool ok = true;
    for (int r = 0; r < runs; r++) {
        ok = fn(A, B, pano, t, st);
        double* ps = &sum.gray; double* pt = &t.gray; double* pm = &mn.gray;
        int nf = sizeof(StageTimes) / sizeof(double);
        for (int k = 0; k < nf; k++) { ps[k] += pt[k]; if (r == 0 || pt[k] < pm[k]) pm[k] = pt[k]; }
    }
    double* ps = &sum.gray; for (size_t k = 0; k < sizeof(StageTimes)/sizeof(double); k++) ps[k] /= runs;

    printf("version=%d  input=%dx%d  runs=%d  device=CPU(single thread)\n", version, A.w, A.h, runs);
    printf("keypoints A/B: %d/%d  matches: %d  inliers: %d  stitched: %s\n", st.kpA, st.kpB, st.matches, st.inliers, ok ? "yes" : "NO");
    printf("H = [% .5f % .5f % .5f; % .5f % .5f % .5f; % .6f % .6f % .6f]\n",
           st.H[0], st.H[1], st.H[2], st.H[3], st.H[4], st.H[5], st.H[6], st.H[7], st.H[8]);
    printf("stage times (ms, mean | min):\n");
    const char* names[] = {"gray","blur","fast","brief","match","ransac","warp","h2d","d2h","total(e2e)"};
    double* pm = &mn.gray;
    for (int k = 0; k < 10; k++) printf("  %-10s %9.3f | %9.3f\n", names[k], ps[k], pm[k]);
    printf("  %-10s %9.3f\n", "kernels", sum.kernels());
    printf("CSV,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", version, A.w, A.h,
           sum.gray, sum.blur, sum.fast, sum.brief, sum.match, sum.ransac, sum.warp, sum.h2d, sum.d2h, sum.total);
    if (ok && !save_image(argv[3], pano)) { fprintf(stderr, "cannot save output\n"); return 1; }
    if (ok) printf("panorama %dx%d -> %s\n", pano.w, pano.h, argv[3]);
    return ok ? 0 : 2;
}
