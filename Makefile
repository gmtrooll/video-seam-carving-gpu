CC       = nvcc
SRC      = cuda_seam_carving.cu
EXEC     = cuda_seam_carving

# ─── Detect GPU architecture ────────────────────────────────────────────
# RTX 3060 = sm_86.  Change if targeting a different GPU.
ARCH     = -arch=sm_86

# ─── OpenCV flags ───────────────────────────────────────────────────────
CVFLAGS  = $(shell pkg-config --cflags opencv4)
CVLIBS   = $(shell pkg-config --libs   opencv4)

# ─── Compiler flags ────────────────────────────────────────────────────
NVFLAGS  = -O3 $(ARCH) -Xcompiler "-Wall" $(CVFLAGS)
LDFLAGS  = $(CVLIBS)

# ═════════════════════════════════════════════════════════════════════════
$(EXEC): $(SRC)
	$(CC) $(NVFLAGS) $(SRC) -o $(EXEC) $(LDFLAGS)

clean:
	rm -f $(EXEC)

.PHONY: clean
