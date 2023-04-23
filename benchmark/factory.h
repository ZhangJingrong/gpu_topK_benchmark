#ifndef FACTORY_H_
#define FACTORY_H_
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "batch_wrapper.h"
#include "cub_topk.cuh"
#include "drtopk_bitonic.cuh"
#include "drtopk_radix.cuh"
#include "faiss_topk.cuh"
#include "grid_select.h"
#include "nv_util.h"
#include "raft_topk.cuh"
#include "sample_select_topk.cuh"



template<typename T, typename idxT>
class Factory {
public:
    using topk_func_t = std::function<void(void* buf,
                                           size_t& buf_size,
                                           const T* in,
                                           int batch_size,
                                           idxT len,
                                           idxT k,
                                           T* out,
                                           idxT* out_idx,
                                           bool greater,
                                           cudaStream_t stream)>;

    Factory();
    Factory(const Factory&) = delete;
    Factory& operator=(const Factory&) = delete;

    topk_func_t create(const std::string& name) const {
        auto it = algos_.find(name);
        if (it != algos_.end()) {
            return it->second;
        }
        else {
            return nullptr;
        }
    }

    bool has_algo(const std::string& name) const {
        auto it = algos_.find(name);
        return it != algos_.end();
    }

    std::vector<std::string> algo_names() const {
        std::vector<std::string> names;
        for (const auto& algo : algos_) {
            names.push_back(algo.first);
        }
        return names;
    }

private:
    std::map<std::string, topk_func_t> algos_;
};


template<typename T, typename idxT>
Factory<T, idxT>::Factory() {
    algos_ = {
        {"raft_radix_11bits_extra_pass", nv::raft_radix_11bits_extra_pass<T, idxT>},
        {"grid_select", nv::grid_select},
        {"cub", MultiStreamBatchWrapper<T, idxT>(nv::cub_topk<T, idxT>, 8)},
        {"faiss_block", nv::faiss_block_select_topk<T, idxT>},
        {"faiss_warp", nv::faiss_warp_select_topk<T, idxT>},
        {"sampleselect",
         BatchWrapper<T, idxT>(nv::sampleselect::sample_select_topk<T, idxT>)},
        {"sampleselect-bucket",
         BatchWrapper<T, idxT>(nv::sampleselect::bucket_select_topk<T, idxT>)},
        {"sampleselect-quick",
         BatchWrapper<T, idxT>(nv::sampleselect::quick_select_topk<T, idxT>)},
        {"drtopk_radix", BatchWrapper<T, idxT>(nv::drtopk::drtopk_radix_topk<T, idxT>)},
        {"drtopk_bitonic",
         BatchWrapper<T, idxT>(nv::drtopk::drtopk_bitonic_topk<T, idxT>)},
    };
}


#endif
