#ifndef CPU_TOPK_H_
#define CPU_TOPK_H_
#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

namespace nv {

template<typename T, typename idxT, typename Compare>
void cpu_topk(
    const T* in, int batch_size, idxT len, idxT k, T* out, idxT* out_idx, Compare cmp) {
    if (!out_idx) {
        for (size_t i = 0; i < static_cast<size_t>(batch_size); ++i) {
            std::partial_sort_copy(
                in + i * len, in + (i + 1) * len, out + i * k, out + (i + 1) * k, cmp);
        }
        return;
    }

    std::vector<idxT> in_idx(len);
    for (size_t i = 0; i < static_cast<size_t>(batch_size); ++i) {
        std::iota(in_idx.begin(), in_idx.end(), 0);
        std::partial_sort_copy(in_idx.cbegin(),
                               in_idx.cend(),
                               out_idx + i * k,
                               out_idx + (i + 1) * k,
                               [in, i, len, cmp](idxT a, idxT b) {
                                   return cmp(in[i * len + a], in[i * len + b]);
                               });
        std::transform(
            out_idx + i * k, out_idx + (i + 1) * k, out + i * k, [in, i, len](idxT j) {
                return in[i * len + j];
            });
    }
}



// has different API from GPU topk
// it uses host pointer rather than device pointer anyway
template<typename T, typename idxT>
void cpu_topk(const T* in,
              int batch_size,
              idxT len,
              idxT k,
              T* out,
              idxT* out_idx = nullptr,
              bool greater = true) {
    if (greater) {
        cpu_topk(in, batch_size, len, k, out, out_idx, std::greater<T>());
    }
    else {
        cpu_topk(in, batch_size, len, k, out, out_idx, std::less<T>());
    }
}


}  // namespace nv
#endif
