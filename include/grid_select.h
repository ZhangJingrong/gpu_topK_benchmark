#ifndef GRID_SELECT_H_
#define GRID_SELECT_H_

#include "cuda_runtime_api.h"

namespace nv {

void grid_select(void* buf,
                 size_t& buf_size,
                 const float* in,
                 int batch_size,
                 int len,
                 int k,
                 float* out,
                 int* out_idx,
                 bool greater,
                 cudaStream_t stream = 0);



}
#endif
