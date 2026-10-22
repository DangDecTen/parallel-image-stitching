# V0 only for now. stb_image.h / stb_image_write.h must be in src/ (or add -DNO_STB to use .ppm only).
CXXFLAGS = -O2 -std=c++17
stitch: src/main.cpp src/v0_seq.cpp src/common.h src/stitch.h src/io.h
	g++ $(CXXFLAGS) src/main.cpp src/v0_seq.cpp -o stitch
clean:
	rm -f stitch
