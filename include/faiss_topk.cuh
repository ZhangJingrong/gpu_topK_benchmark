#ifndef FAISS_TOPK_CUH_
#define FAISS_TOPK_CUH_
#include "faiss/gpu/utils/BlockSelectKernel.cuh"
#include "faiss/gpu/utils/WarpSelectKernel.cuh"


namespace nv {

template<typename T, typename idxT>
void faiss_warp_select_topk(void* buf,
                            size_t& buf_size,
                            const T* in,
                            int batch_size,
                            idxT len,
                            idxT k,
                            T* out,
                            idxT* out_idx = nullptr,
                            bool greater = true,
                            cudaStream_t stream = 0) {
    if (!buf) {
        buf_size = 1;
        return;
    }

    idxT in_dim[2] = {batch_size, len};
    idxT out_dim[2] = {batch_size, k};
    faiss::gpu::DeviceTensor<T, 2, true, idxT> in_tensor(const_cast<T*>(in), in_dim);
    faiss::gpu::DeviceTensor<T, 2, true, idxT> out_tensor(out, out_dim);
    faiss::gpu::DeviceTensor<idxT, 2, true, idxT> out_idx_tensor(out_idx, out_dim);
    faiss::gpu::runWarpSelect(in_tensor, out_tensor, out_idx_tensor, greater, k, stream);
}



template<typename T, typename idxT>
void faiss_block_select_topk(void* buf,
                             size_t& buf_size,
                             const T* in,
                             int batch_size,
                             idxT len,
                             idxT k,
                             T* out,
                             idxT* out_idx = nullptr,
                             bool greater = true,
                             cudaStream_t stream = 0) {
    if (!buf) {
        buf_size = 1;
        return;
    }

    idxT in_dim[2] = {batch_size, len};
    idxT out_dim[2] = {batch_size, k};
    faiss::gpu::DeviceTensor<T, 2, true, idxT> in_tensor(const_cast<T*>(in), in_dim);
    faiss::gpu::DeviceTensor<T, 2, true, idxT> out_tensor(out, out_dim);
    faiss::gpu::DeviceTensor<idxT, 2, true, idxT> out_idx_tensor(out_idx, out_dim);
    faiss::gpu::runBlockSelect(in_tensor, out_tensor, out_idx_tensor, greater, k, stream);
}



}  // namespace nv
#endif
