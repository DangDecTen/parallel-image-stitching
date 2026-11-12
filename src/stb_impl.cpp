// stb_impl.cpp - compiles the stb single-header libraries exactly once (keeps io.h / main.cpp fast to build).
// Authors: <TODO: Ten - MSSV, ...>
// External sources: stb_image.h, stb_image_write.h (Sean Barrett, public domain) - https://github.com/nothings/stb
#ifndef NO_STB
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#endif
