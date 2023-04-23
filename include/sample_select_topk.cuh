#ifndef SAMPLE_SELECT_TOPK_CUH_
#define SAMPLE_SELECT_TOPK_CUH_
#include "filter.cuh"
#include "kernel_config.cuh"
#include "launcher_fwd.cuh"
#include "nv_util.h"

namespace nv {
namespace sampleselect {

constexpr auto max_tree_width = 4096;
constexpr auto max_tree_size = 2 * max_tree_width * 2;
constexpr auto max_block_count = 1024;
enum class Algo {
    SAMPLE_SELECT,
    BUCKET_SELECT,
    QUICK_SELECT,
};


template<typename T, typename idxT>
void sample_select_topk_(Algo algo,
                         void* buf,
                         size_t& buf_size,
                         const T* in,
                         idxT len,
                         idxT k,
                         T* out,
                         idxT* out_idx,
                         bool greater,
                         cudaStream_t stream) {
    static_assert(std::is_same<idxT, gpu::index>::value,
                  "idxT is not the same as gpu::index");
    assert(greater == false);

    T* in_copy;
    T* tmp;
    T* tree;
    gpu::index* count_tmp;
    T* kth_value;
    idxT* counter;

    std::vector<size_t> sizes = {
        len * sizeof(*in_copy),
        len * sizeof(*tmp),
        max_tree_size * sizeof(*tree),
        // from app/test_fixture.cuh:
        (len + max_block_count * max_tree_width * 16) * sizeof(*count_tmp),
        sizeof(*kth_value),
        sizeof(*counter)};

    size_t total_size = calc_aligned_size(sizes);
    if (!buf) {
        buf_size = total_size;
        return;
    }
    std::vector<void*> aligned_pointers = calc_aligned_pointers(buf, sizes);
    in_copy = static_cast<T*>(aligned_pointers[0]);
    tmp = static_cast<T*>(aligned_pointers[1]);
    tree = static_cast<T*>(aligned_pointers[2]);
    count_tmp = static_cast<gpu::index*>(aligned_pointers[3]);
    kth_value = static_cast<T*>(aligned_pointers[4]);
    counter = static_cast<idxT*>(aligned_pointers[5]);

    TOPK_CUDA_CHECK(cudaMemset(counter, 0, sizeof(*counter)));
    TOPK_CUDA_CHECK(
        cudaMemcpy(in_copy, in, len * sizeof(*in_copy), cudaMemcpyDeviceToDevice));

    gpu::index rank = k - 1;
    if (algo == Algo::SAMPLE_SELECT) {
        gpu::sampleselect<T, gpu::select_config<>>(
            in_copy, tmp, tree, count_tmp, (gpu::index)len, rank, kth_value);
    }
    else if (algo == Algo::BUCKET_SELECT) {
        gpu::sampleselect<
            T,
            gpu::select_config<10, 10, 8, true, true, true, 8, 10, 10, true>>(
            in_copy, tmp, tree, count_tmp, (gpu::index)len, rank, kth_value);
    }
    else if (algo == Algo::QUICK_SELECT) {
        gpu::quickselect<T, gpu::select_config<>>(
            in_copy, tmp, count_tmp, (gpu::index)len, rank, kth_value);
    }

    filter(in, len, k, kth_value, counter, out, out_idx, greater, stream);
}


template<typename T, typename idxT>
void sample_select_topk(void* buf,
                        size_t& buf_size,
                        const T* in,
                        idxT len,
                        idxT k,
                        T* out,
                        idxT* out_idx = nullptr,
                        bool greater = true,
                        cudaStream_t stream = 0) {
    sample_select_topk_(
        Algo::SAMPLE_SELECT, buf, buf_size, in, len, k, out, out_idx, greater, stream);
}


template<typename T, typename idxT>
void bucket_select_topk(void* buf,
                        size_t& buf_size,
                        const T* in,
                        idxT len,
                        idxT k,
                        T* out,
                        idxT* out_idx = nullptr,
                        bool greater = true,
                        cudaStream_t stream = 0) {
    sample_select_topk_(
        Algo::BUCKET_SELECT, buf, buf_size, in, len, k, out, out_idx, greater, stream);
}


template<typename T, typename idxT>
void quick_select_topk(void* buf,
                       size_t& buf_size,
                       const T* in,
                       idxT len,
                       idxT k,
                       T* out,
                       idxT* out_idx = nullptr,
                       bool greater = true,
                       cudaStream_t stream = 0) {
    sample_select_topk_(
        Algo::QUICK_SELECT, buf, buf_size, in, len, k, out, out_idx, greater, stream);
}




}  // namespace sampleselect
}  // namespace nv
#endif
