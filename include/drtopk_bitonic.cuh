#ifndef DRTOPK_BITONIC_CUH_
#define DRTOPK_BITONIC_CUH_

#include "bitonic_com.cuh"
#include "nv_util.h"
namespace nv {
namespace drtopk {

template<typename T, typename idxT>
void drtopk_bitonic_topk(void* buf,
                         size_t& buf_size,
                         const T* in,
                         idxT len,
                         idxT k,
                         T* out,
                         idxT* out_idx = nullptr,
                         bool greater = true,
                         cudaStream_t stream = 0) {
    int alpha = 11;
    idxT SubRangesize = pow(2, alpha);
    idxT NSubranges = len / SubRangesize;

    assert(len < 33554432);
    T* in_copy;
    T* Max_d;
    idxT* SubrangeId_d;
    T* ConcatenatedRange_d;


    std::vector<size_t> sizes = {len * sizeof(*in_copy),
                                 sizeof(T) * NSubranges,
                                 sizeof(idxT) * NSubranges,
                                 sizeof(T) * k * SubRangesize};

    size_t total_size = calc_aligned_size(sizes);

    if (!buf) {
        buf_size = total_size;
        return;
    }
    std::vector<void*> aligned_pointers = calc_aligned_pointers(buf, sizes);
    in_copy = static_cast<T*>(aligned_pointers[0]);
    Max_d = static_cast<T*>(aligned_pointers[1]);
    SubrangeId_d = static_cast<idxT*>(aligned_pointers[2]);
    ConcatenatedRange_d = static_cast<T*>(aligned_pointers[3]);

    idxT reverse_counter = k - 1;
    TOPK_CUDA_CHECK(cudaMemset(out_idx, -1, sizeof(idxT) * k));
    TOPK_CUDA_CHECK(cudaMemset(out, 0, sizeof(idxT) * k));
    TOPK_CUDA_CHECK(cudaMemcpy(in_copy, in, len * sizeof(T), cudaMemcpyDeviceToDevice));

    T TopKElement;
    if (NSubranges > k) {
        int Nthreadstowork = 32;
        if (SubRangesize < 32) {
            Nthreadstowork = SubRangesize;
        }
        drtopk_bitonic::sampleMax<T, idxT><<<128, 128>>>(in_copy,
                                                         Max_d,
                                                         len,
                                                         NSubranges,
                                                         SubRangesize,
                                                         alpha,
                                                         SubrangeId_d,
                                                         Nthreadstowork);
        TOPK_CUDA_CHECK(cudaDeviceSynchronize());
        drtopk_bitonic::bitonic_firstTopk<T, idxT>(Max_d,
                                                   NSubranges,
                                                   k,
                                                   SubRangesize,
                                                   NSubranges,
                                                   SubrangeId_d,
                                                   in_copy,
                                                   ConcatenatedRange_d);
        TOPK_CUDA_CHECK(cudaDeviceSynchronize());
        drtopk_bitonic::bitonic<T, idxT>(ConcatenatedRange_d,
                                         k * SubRangesize,
                                         k,
                                         TopKElement,
                                         SubRangesize,
                                         nullptr,
                                         len,
                                         SubrangeId_d,
                                         in_copy,
                                         ConcatenatedRange_d);
        TOPK_CUDA_CHECK(cudaDeviceSynchronize());
    }
    else {
        drtopk_bitonic::bitonic<T, idxT>(in_copy,
                                         len,
                                         k,
                                         TopKElement,
                                         SubRangesize,
                                         nullptr,
                                         len,
                                         SubrangeId_d,
                                         in_copy,
                                         ConcatenatedRange_d);
        TOPK_CUDA_CHECK(cudaDeviceSynchronize());
    }
    TOPK_CUDA_CHECK(cudaDeviceSynchronize());
    TOPK_CUDA_CHECK(cudaMemcpy(out, in_copy, sizeof(T) * k, cudaMemcpyDeviceToDevice));
}



}  // namespace drtopk
}  // namespace nv
#endif