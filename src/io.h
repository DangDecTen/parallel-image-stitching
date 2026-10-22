// io.h - image loading/saving. PNG/JPG through stb (download stb_image.h / stb_image_write.h
// from https://github.com/nothings/stb into src/), binary PPM (P6) is built in.
#pragma once
#include "common.h"
#include <cstdio>
#include <cstring>
#include <string>
#ifndef NO_STB
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#endif

static bool has_ext(const std::string& p, const char* e) {
    size_t n = strlen(e); return p.size() >= n && p.compare(p.size() - n, n, e) == 0; }

static bool load_image(const std::string& path, Image& im) {
    if (has_ext(path, ".ppm")) {
        FILE* f = fopen(path.c_str(), "rb"); if (!f) return false;
        char magic[3]; int w, h, mx;
        if (fscanf(f, "%2s %d %d %d", magic, &w, &h, &mx) != 4 || strcmp(magic, "P6")) { fclose(f); return false; }
        fgetc(f);
        im.w = w; im.h = h; im.rgb.resize((size_t)w * h * 3);
        size_t r = fread(im.rgb.data(), 1, im.rgb.size(), f); fclose(f);
        return r == im.rgb.size();
    }
#ifndef NO_STB
    int w, h, c; unsigned char* d = stbi_load(path.c_str(), &w, &h, &c, 3);
    if (!d) return false;
    im.w = w; im.h = h; im.rgb.assign(d, d + (size_t)w * h * 3); stbi_image_free(d);
    return true;
#else
    return false;
#endif
}

static bool save_image(const std::string& path, const Pano& p) {
    if (has_ext(path, ".ppm")) {
        FILE* f = fopen(path.c_str(), "wb"); if (!f) return false;
        fprintf(f, "P6\n%d %d\n255\n", p.w, p.h); fwrite(p.rgb.data(), 1, p.rgb.size(), f); fclose(f); return true;
    }
#ifndef NO_STB
    if (has_ext(path, ".png")) return stbi_write_png(path.c_str(), p.w, p.h, 3, p.rgb.data(), p.w * 3) != 0;
    return stbi_write_jpg(path.c_str(), p.w, p.h, 3, p.rgb.data(), 95) != 0;
#else
    return false;
#endif
}
