GENCODES = -arch=sm_80

CPPFLAGS += -DNDEBUG \
            -I../include \
            -I../third_party/faiss \
            -isystem ../third_party/DrTopKSC/bitonic/LargerKVersions/largerK/ \
            -I../third_party/DrTopKSC/baseline+filter+beta+shuffle/ \
            -I../third_party/gpu_selection/include -I../third_party/gpu_selection/lib \
            -isystem ../third_party/raft/cpp/include \
            -isystem ../third_party/rmm/include \
            -I../third_party/spdlog/include \
            -I../third_party

LDFLAGS += -L../third_party/faiss -lfaiss -Xlinker -rpath=../third_party/faiss \
           -L../third_party/gpu_selection -lgpu_selection -Xlinker -rpath=../third_party/gpu_selection/ \
           -L../third_party -lgridselect -Xlinker -rpath=../third_party \
           -lcurand

NVCCFLAGS = -c -O2 -std=c++17 \
            -Xcompiler "-Wall -Wextra -Wno-unused-parameter" \
            --expt-relaxed-constexpr \
            --extended-lambda


benchmark: benchmark.o
	nvcc $(GENCODES) $(LDFLAGS) -o $@ $<

benchmark.o: benchmark.cu $(wildcard *.cuh *.h ../include/*.cuh ../include/*.h)
	nvcc $(NVCCFLAGS) $(GENCODES) $(CPPFLAGS) -o $@ $<




.PHONY: clean
clean:
	rm benchmark benchmark.o
