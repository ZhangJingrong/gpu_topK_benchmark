#ifndef RAFT_TOPK_CUH_
#define RAFT_TOPK_CUH_
#include "raft/matrix/detail/select_radix.cuh"
#include "rmm/mr/device/per_device_resource.hpp"
#include "rmm/mr/device/pool_memory_resource.hpp"


namespace nv {

namespace {

// https://github.com/rapidsai/raft/blob/branch-22.06/cpp/bench/common/benchmark.hpp
struct using_pool_memory_res {
private:
    rmm::mr::device_memory_resource* orig_res_;
    rmm::mr::cuda_memory_resource cuda_res_;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_res_;

public:
    using_pool_memory_res(size_t initial_size, size_t max_size)
        : orig_res_(rmm::mr::get_current_device_resource()),
          pool_res_(&cuda_res_, initial_size, max_size) {
        rmm::mr::set_current_device_resource(&pool_res_);
    }

    using_pool_memory_res()
        : using_pool_memory_res(size_t(1) << size_t(30), size_t(16) << size_t(30)) {}

    ~using_pool_memory_res() { rmm::mr::set_current_device_resource(orig_res_); }
};
using_pool_memory_res rmm_res;

}  // namespace



template<typename T, typename idxT>
void raft_radix_11bits(void* buf,
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

    raft::matrix::detail::select::radix::select_k<T, idxT, 11, 512>(
        in,
        static_cast<idxT*>(nullptr),
        batch_size,
        len,
        k,
        out,
        out_idx,
        !greater,
        true,  // fused_last_filter
        stream);
}


template<typename T, typename idxT>
void raft_radix_11bits_extra_pass(void* buf,
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

    raft::matrix::detail::select::radix::select_k<T, idxT, 11, 512>(
        in,
        static_cast<idxT*>(nullptr),
        batch_size,
        len,
        k,
        out,
        out_idx,
        !greater,
        false,  // fused_last_filter
        stream);
}


}  // namespace nv
#endif
