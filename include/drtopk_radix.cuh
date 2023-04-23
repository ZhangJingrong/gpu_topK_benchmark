#ifndef DRTOPK_RADIX_CUH_
#define DRTOPK_RADIX_CUH_
#include <cstring>

#include "filter.cuh"
#include "nv_util.h"
#include "radixselect.cuh"

namespace nv {
namespace drtopk {

float get_value(unsigned int f) {
    unsigned int mask = ((f >> 31) - 1) | 0x80000000;
    mask = f ^ mask;
    float value;
    memcpy(&value, &mask, sizeof(value));
    return value;
}



template<typename data_t, typename index_t>
void radix_select_inplace_out(const data_t* vec_d,
                              data_t* out,
                              index_t* out_idx,
                              data_t* kth_value,
                              index_t* counter,
                              index_t* reverse_counter,
                              index_t num_element,
                              index_t k,
                              index_t num_bucket,
                              data_t& TopKElement,
                              int NBitsperDigit,
                              int CurrentDigit,
                              unsigned int flag) {
    index_t* Count =
        (index_t*)malloc(sizeof(index_t) * num_bucket);  // new index_t[num_bucket];
    index_t* Count_d;
    TOPK_CUDA_CHECK(cudaMalloc((void**)&Count_d, sizeof(index_t) * num_bucket));

    index_t* CumCount =
        (index_t*)malloc(sizeof(index_t) * num_bucket);  // new index_t[num_bucket];

    index_t Belowcount = 0;
    int Kdigit = 0;

    float f_flag = 0;
    while (CurrentDigit >= 0) {
        TOPK_CUDA_CHECK(cudaMemset(Count_d, 0, num_bucket * sizeof(index_t)));
        int shleft = (CurrentDigit + 1) * (NBitsperDigit);
        int shright = CurrentDigit * NBitsperDigit;

        drtopk_radix::CalculateOccurence_inplace<data_t, index_t>
            <<<128, 128, num_bucket * sizeof(index_t)>>>(vec_d,
                                                         out,
                                                         k,
                                                         num_element,
                                                         Count_d,
                                                         NBitsperDigit,
                                                         CurrentDigit,
                                                         num_bucket,
                                                         flag,
                                                         shleft,
                                                         shright);
        TOPK_CUDA_CHECK(cudaDeviceSynchronize());

        TOPK_CUDA_CHECK(cudaMemcpy(
            Count, Count_d, sizeof(index_t) * num_bucket, cudaMemcpyDeviceToHost));
        memset(CumCount, 0, num_bucket * sizeof(index_t));
        drtopk_radix::CumulateCount_inplace<data_t, index_t>(Count,
                                                             CumCount,
                                                             num_bucket,
                                                             Kdigit,
                                                             k,
                                                             num_element,
                                                             Belowcount,
                                                             flag,
                                                             NBitsperDigit,
                                                             CurrentDigit);
        if (Kdigit != 0) Belowcount = CumCount[Kdigit - 1];
        f_flag = get_value(flag);
        CurrentDigit--;
    }

    TOPK_CUDA_CHECK(
        cudaMemcpy(kth_value, &f_flag, sizeof(data_t), cudaMemcpyHostToDevice));
    nv::filter<data_t, index_t>(vec_d,
                                1,
                                num_element,
                                k,
                                kth_value,
                                counter,
                                reverse_counter,
                                out,
                                out_idx,
                                true,
                                0);
}

template<typename T, typename idxT>
void drtopk_radix_topk(void* buf,
                       size_t& buf_size,
                       const T* in,
                       idxT len,
                       idxT k,
                       T* out,
                       idxT* out_idx = nullptr,
                       bool greater = true,
                       cudaStream_t stream = 0) {
    T* d_k_value = nullptr;
    idxT* d_counter = nullptr;
    idxT* d_reverse_counter = nullptr;
    std::vector<size_t> sizes = {sizeof(T), sizeof(idxT), sizeof(idxT)};
    size_t total_size = calc_aligned_size(sizes);
    if (!buf) {
        buf_size = total_size;
        return;
    }
    std::vector<void*> aligned_pointers = calc_aligned_pointers(buf, sizes);
    d_k_value = static_cast<T*>(aligned_pointers[0]);
    d_counter = static_cast<idxT*>(aligned_pointers[1]);
    d_reverse_counter = static_cast<idxT*>(aligned_pointers[2]);

    idxT reverse_counter = k - 1;
    TOPK_CUDA_CHECK(cudaMemcpy(
        d_reverse_counter, &reverse_counter, sizeof(idxT), cudaMemcpyHostToDevice));
    TOPK_CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(idxT)));
    TOPK_CUDA_CHECK(cudaMemset(out_idx, -1, sizeof(idxT) * k));
    TOPK_CUDA_CHECK(cudaMemset(out, 0, sizeof(idxT) * k));

    T TopKElement = 0;
    const int NBits = 8;
    idxT numBuckets = 1 << NBits;
    int CurrentDigit = (sizeof(T) * 8 / NBits) - 1;
    unsigned int flag = 0;

    radix_select_inplace_out<T, idxT>(in,
                                      out,
                                      out_idx,
                                      d_k_value,
                                      d_counter,
                                      d_reverse_counter,
                                      len,
                                      k,
                                      numBuckets,
                                      TopKElement,
                                      NBits,
                                      CurrentDigit,
                                      flag);
}

}  // namespace drtopk
}  // namespace nv
#endif
