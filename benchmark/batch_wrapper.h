#ifndef BATCH_WRAPPER_H_
#define BATCH_WRAPPER_H_
#include <functional>
#include <vector>
#include "nv_util.h"

template<typename T, typename idxT>
class BatchWrapper {
private:
    using func_t = std::function<void(void* buf,
                                      size_t& buf_size,
                                      const T* in,
                                      idxT len,
                                      idxT k,
                                      T* out,
                                      idxT* out_idx,
                                      bool greater,
                                      cudaStream_t stream)>;

public:
    explicit BatchWrapper(func_t f): f_(f) {}
    void operator()(void* buf,
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
            f_(nullptr, buf_size, in, len, k, out, out_idx, greater, stream);
            return;
        }
        for (int i = 0; i < batch_size; ++i) {
            f_(buf,
               buf_size,
               in + i * len,
               len,
               k,
               out + i * k,
               out_idx + i * k,
               greater,
               stream);
        }
    }

private:
    func_t f_;
};



template<typename T, typename idxT>
class MultiStreamBatchWrapper {
private:
    using func_t = std::function<void(void* buf,
                                      size_t& buf_size,
                                      const T* in,
                                      idxT len,
                                      idxT k,
                                      T* out,
                                      idxT* out_idx,
                                      bool greater,
                                      cudaStream_t stream)>;

public:
    MultiStreamBatchWrapper(func_t f, int n_stream): f_(f), n_stream_(n_stream) {
        // max number of concurrent kernels is 128 for CC <= 8.0
        assert(n_stream_ > 0 && n_stream_ <= 128);
        create_streams_and_events_();
    }

    // need copy ctor so it can be put in a container
    MultiStreamBatchWrapper(const MultiStreamBatchWrapper& other)
        : f_(other.f_), n_stream_(other.n_stream_) {
        // each object has its own set of streams and events
        create_streams_and_events_();
    }

    MultiStreamBatchWrapper& operator=(const MultiStreamBatchWrapper&) = delete;

    ~MultiStreamBatchWrapper() {
        for (auto& stream : streams_) {
            TOPK_CUDA_CHECK(cudaStreamDestroy(stream));
        }
        for (auto& event : events_) {
            TOPK_CUDA_CHECK(cudaEventDestroy(event));
        }
    }

    void operator()(void* buf,
                    size_t& buf_size,
                    const T* in,
                    int batch_size,
                    idxT len,
                    idxT k,
                    T* out,
                    idxT* out_idx = nullptr,
                    bool greater = true,
                    cudaStream_t stream = 0) {
        size_t sub_buf_size;
        f_(nullptr, sub_buf_size, in, len, k, out, out_idx, greater, stream);
        std::vector<size_t> sizes(n_stream_, sub_buf_size);

        if (!buf) {
            buf_size = nv::calc_aligned_size(sizes);
            return;
        }
        std::vector<void*> sub_bufs = nv::calc_aligned_pointers(buf, sizes);

        for (int i = 0; i < batch_size; ++i) {
            f_(sub_bufs[i % n_stream_],
               sub_buf_size,
               in + i * len,
               len,
               k,
               out + i * k,
               out_idx + i * k,
               greater,
               streams_[i % n_stream_]);
        }
        for (int i = 0; i < n_stream_ && i < batch_size; ++i) {
            TOPK_CUDA_CHECK(cudaEventRecord(events_[i], streams_[i]));
            TOPK_CUDA_CHECK(cudaStreamWaitEvent(stream, events_[i], 0));
        }
    }

private:
    void create_streams_and_events_() {
        streams_.resize(n_stream_);
        // don't bother with exception safety
        for (auto& stream : streams_) {
            TOPK_CUDA_CHECK(cudaStreamCreate(&stream));
        }

        events_.resize(n_stream_);
        for (auto& event : events_) {
            TOPK_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        }
    }

    func_t f_;
    int n_stream_;
    std::vector<cudaStream_t> streams_;
    std::vector<cudaEvent_t> events_;
};



#endif
