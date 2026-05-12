CC       = nvcc
ARCH     = -arch=sm_86

# OpenCV configuration
CVFLAGS  = $(shell pkg-config --cflags opencv4)
CVLIBS   = $(shell pkg-config --libs   opencv4)

# Compiler flags
NVFLAGS  = -O3 $(ARCH) -Xcompiler "-Wall" $(CVFLAGS)
LDFLAGS  = $(CVLIBS)

# Sources
SHARED_SRCS = $(wildcard kernels/shared/*.cu kernels/shared/*.h)
V2_SRCS     = $(wildcard kernels/v2/*.cu)

all: v2_jnd

v2_jnd: v2_jnd.cu $(SHARED_SRCS) $(V2_SRCS)
	$(CC) $(NVFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f v2_jnd

.PHONY: all clean
