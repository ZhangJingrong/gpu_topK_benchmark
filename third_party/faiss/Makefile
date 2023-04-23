CUDA_PATH := /usr/local/cuda
GENCODES = -arch=sm_80

NVCC := $(CUDA_PATH)/bin/nvcc
CPPFLAGS = -DNDEBUG -I. -I$(CUDA_PATH)/include
CXXFLAGS = -std=c++14 -O2 -fPIC
NVCCFLAGS = -std=c++14 -O2 -Xcompiler -fPIC $(GENCODES)


%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -c $< -o $@

objs := \
faiss/gpu/GpuResources.o \
faiss/gpu/utils/DeviceUtils.o \
faiss/gpu/utils/BlockSelectFloat.o \
faiss/gpu/utils/WarpSelectFloat.o \
faiss/gpu/utils/blockselect/BlockSelectFloat128.o \
faiss/gpu/utils/blockselect/BlockSelectFloat1.o \
faiss/gpu/utils/blockselect/BlockSelectFloat256.o \
faiss/gpu/utils/blockselect/BlockSelectFloat32.o \
faiss/gpu/utils/blockselect/BlockSelectFloat64.o \
faiss/gpu/utils/blockselect/BlockSelectFloatF1024.o \
faiss/gpu/utils/blockselect/BlockSelectFloatF2048.o \
faiss/gpu/utils/blockselect/BlockSelectFloatF512.o \
faiss/gpu/utils/blockselect/BlockSelectFloatT1024.o \
faiss/gpu/utils/blockselect/BlockSelectFloatT2048.o \
faiss/gpu/utils/blockselect/BlockSelectFloatT512.o \
faiss/gpu/utils/warpselect/WarpSelectFloat128.o \
faiss/gpu/utils/warpselect/WarpSelectFloat1.o \
faiss/gpu/utils/warpselect/WarpSelectFloat256.o \
faiss/gpu/utils/warpselect/WarpSelectFloat32.o \
faiss/gpu/utils/warpselect/WarpSelectFloat64.o \
faiss/gpu/utils/warpselect/WarpSelectFloatF1024.o \
faiss/gpu/utils/warpselect/WarpSelectFloatF2048.o \
faiss/gpu/utils/warpselect/WarpSelectFloatF512.o \
faiss/gpu/utils/warpselect/WarpSelectFloatT1024.o \
faiss/gpu/utils/warpselect/WarpSelectFloatT2048.o \
faiss/gpu/utils/warpselect/WarpSelectFloatT512.o \


all: libfaiss.so 

libfaiss.so: $(objs)
	$(NVCC) -o $@ $(NVCCFLAGS) -Xcompiler "-fPIC" -Xcompiler "-shared" $^ -lcublas

.PHONY: clean
clean:
	-rm $(objs) libfaiss.so
