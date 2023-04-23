# Top-K benchmark on GPU

## Tested Environment
* Ubuntu 22.04
* CUDA 12.0
* NVIDIA A100 GPU


## Build

### Dependencies
* RAFT (https://github.com/rapidsai/raft)
* Faiss (https://github.com/facebookresearch/faiss)
* gpu_selection (https://github.com/upsj/gpu_selection)
* DrTopK (https://github.com/Anil-Gaihre/DrTopKSC)

A script is provided to get them:
```bash
cd third_party && ./download.sh
```

### build Faiss
```bash
cd third_party/faiss && make
```

### build gpu_selection
1) The default index type in gpu_selection is `uint32_t`, however, this benchmark uses `int`. The index type defined in gpu_selection/include/cuda_definitions.cuh should be modified. Change the first three `using` lines to
```
using index = int;
using poracle = int;
using oracle = int;
```
2) Add a line `#include <limits>` in gpu_selection/lib/verification.cpp.

3) Consider changing gpu_selection/CMakeLists.txt. Delete the last line and change the fourth line to
```
list(APPEND CMAKE_CUDA_FLAGS "-arch=sm_80 -rdc=true --maxrregcount 64")
option(BUILD_SHARED_LIBS "" ON)
```

Cmake is required to buid gpu_selection:
```bash
cd third_party/gpu_selection
mkdir build
cd build
cmake ..
make
mv lib/libgpu_selection.so ..
```

### build benchmark

After building the dependencies, run `cd benchmark && make` to build the benchmark.


## Benchmark Usage
Run `./benchmark` without any argument to see the usage and available algorithms.

Use `-c` to check correctness. Use `-n` with a large number (e.g. `-n 100`) to get more stable benchmark results. Use `-w` to set the number of warmup runs.

For example, to run algorithm CUB for batch_size=1, len=1e6, k=20 (note that exponential form is acceptable for batch/len/k):
```bash
$ ./benchmark -c -w 20 -n 100 cub 1 1e6 20
```
