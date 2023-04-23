#ifndef TEST_UTIL_H_
#define TEST_UTIL_H_
#include <curand.h>
#include <stdio.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "nv_util.h"

#define CURAND_CALL(x)                                                     \
    do {                                                                   \
        curandStatus ret = (x);                                            \
        if (ret != CURAND_STATUS_SUCCESS) {                                \
            printf("cuRAND Error %d at %s:%d\n", ret, __FILE__, __LINE__); \
            exit(1);                                                       \
        }                                                                  \
    } while (0)



template<typename T, typename = std::enable_if_t<sizeof(T) == sizeof(unsigned)>>
__global__ void mask_high_bits(T* data, size_t len, int num_random_bits) {
    unsigned mask = (1u << num_random_bits) - 1;
    unsigned high_bits;
    {
        T t(1);  // generated data will be around this value
        memcpy(&high_bits, &t, sizeof(T));
        high_bits &= ~mask;
    }

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = tid; i < len; i += blockDim.x * gridDim.x) {
        unsigned bits;
        memcpy(&bits, &data[i], sizeof(T));
        bits = high_bits | (bits & mask);
        memcpy(&data[i], &bits, sizeof(T));
    }

    // for stricter correctness validation, add data that hasn't the same prefix bits
    if (tid == 0 && len > 4) {
        unsigned min_bits = high_bits;
        unsigned max_bits = high_bits | mask;
        --min_bits;
        ++max_bits;
        memcpy(&data[0], &min_bits, sizeof(T));
        memcpy(&data[1], &max_bits, sizeof(T));
        data[2] = std::numeric_limits<T>::lowest();
        data[3] = std::numeric_limits<T>::max();
    }
}

// not consider alignment
// so should only be used in TestData, in which alignment is assured by cudaMalloc()
template<typename T>
void fill_random_bits(T* data, size_t len, curandGenerator_t& gen, int num_random_bits) {
    assert(num_random_bits > 0 && num_random_bits <= sizeof(T) * 8);

    // curandGenerate() produces bit randomness, bits beyond casted_len are not filled
    size_t casted_len = sizeof(T) * len / sizeof(unsigned);
    CURAND_CALL(curandGenerate(gen, reinterpret_cast<unsigned*>(data), casted_len));

    if (num_random_bits != sizeof(T) * 8) {
        constexpr int block_dim = 512;
        mask_high_bits<<<(len - 1) / block_dim + 1, block_dim>>>(
            data, len, num_random_bits);
        TOPK_CUDA_CHECK_LAST_ERROR();
    }
}

template<typename T>
void fill_test_data(
    T* data, size_t len, curandGenerator_t& gen, bool, int num_random_bits) {
    fill_random_bits(data, len, gen, num_random_bits);
}

template<>
void fill_test_data(
    float* data, size_t len, curandGenerator_t& gen, bool gaussian, int num_random_bits) {
    if (num_random_bits != sizeof(float) * 8) {
        fill_random_bits(data, len, gen, num_random_bits);
    }
    else if (gaussian) {
        assert(len % 2 == 0);
        CURAND_CALL(curandGenerateNormal(gen, data, len, 0, 1.0f));
    }
    else {
        CURAND_CALL(curandGenerateUniform(gen, data, len));
    }
}




template<typename T, typename idxT>
class TestData {
public:
    T* h_in;
    T* h_out;
    idxT* h_out_idx;
    T* h_out_truth;
    idxT* h_out_idx_truth;
    T* d_in;
    T* d_out;
    idxT* d_out_idx;

    TestData(int batch_size,
             idxT len,
             idxT k,
             bool gaussian,
             int num_random_bits = sizeof(T) * 8)
        : len_(batch_size * len),
          k_(batch_size * k),
          num_random_bits_(num_random_bits),
          gaussian_(gaussian) {
        TOPK_CUDA_CHECK(cudaStreamCreate(&stream_));
        CURAND_CALL(curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen_, std::random_device{}()));
        CURAND_CALL(curandSetStream(gen_, stream_));

        alloc_mem_();
        fill_(d_in, batch_size, len);
        TOPK_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    TestData(const TestData&) = delete;
    TestData& operator=(const TestData&) = delete;

    ~TestData() {
        free_mem_();
        CURAND_CALL(curandDestroyGenerator(gen_));
        TOPK_CUDA_CHECK(cudaStreamDestroy(stream_));
    }

    void d2h_in() {
        TOPK_CUDA_CHECK(cudaMemcpy(h_in, d_in, len_ * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void d2h_out() {
        TOPK_CUDA_CHECK(cudaMemcpy(h_out, d_out, k_ * sizeof(T), cudaMemcpyDeviceToHost));
        TOPK_CUDA_CHECK(
            cudaMemcpy(h_out_idx, d_out_idx, k_ * sizeof(idxT), cudaMemcpyDeviceToHost));
    }

private:
    void alloc_mem_() {
        h_in = (T*)malloc(len_ * sizeof(T));
        h_out = (T*)malloc(k_ * sizeof(T));
        h_out_idx = (idxT*)malloc(k_ * sizeof(idxT));
        h_out_truth = (T*)malloc(k_ * sizeof(T));
        h_out_idx_truth = (idxT*)malloc(k_ * sizeof(idxT));
        assert(h_in);
        assert(h_out);
        assert(h_out_idx);
        assert(h_out_truth);
        assert(h_out_idx_truth);

        TOPK_CUDA_CHECK(cudaMalloc((void**)&d_in, len_ * sizeof(T)));
        TOPK_CUDA_CHECK(cudaMalloc((void**)&d_out, k_ * sizeof(T)));
        TOPK_CUDA_CHECK(cudaMalloc((void**)&d_out_idx, k_ * sizeof(idxT)));
    }

    void free_mem_() {
        free(h_in);
        free(h_out);
        free(h_out_idx);
        free(h_out_truth);
        free(h_out_idx_truth);
        TOPK_CUDA_CHECK(cudaFree(d_in));
        TOPK_CUDA_CHECK(cudaFree(d_out));
        TOPK_CUDA_CHECK(cudaFree(d_out_idx));
    }

    void fill_(T* data, idxT batch_size, idxT len) {
        fill_test_data(data, batch_size * len, gen_, gaussian_, num_random_bits_);
    }


    idxT len_;
    idxT k_;
    cudaStream_t stream_;
    curandGenerator_t gen_;
    int num_random_bits_;
    bool gaussian_;
};



// workaround: no easy way to define host operator!= for half, so use NotEqual instead
template<typename T>
struct NotEqual {
    bool operator()(const T& lhs, const T& rhs) const { return lhs != rhs; }
};


namespace {
class CheckException : public std::runtime_error {
public:
    CheckException(const std::string& what)
        : runtime_error("ERROR [" __FILE__ "::check_result()]: " + what) {}
};
}  // namespace

template<typename T, typename idxT>
void check_result(const T* in,
                  idxT len,
                  const T* v1,
                  const idxT* idx1,
                  const T* v2,
                  const idxT* idx2,
                  idxT k) {
    NotEqual<T> neq;

    if (idx1 && idx2) {
        for (idxT i = 0; i < k; ++i) {
            if (idx1[i] >= len) {
                std::cerr << "len=" << len << ", idx1[" << i << "]=" << idx1[i]
                          << std::endl;
                throw CheckException("idx1[] >= len");
            }
            if (idx2[i] >= len) {
                std::cerr << "len=" << len << ", idx2[" << i << "]=" << idx2[i]
                          << std::endl;
                throw CheckException("idx2[] >= len");
            }
        }

        for (idxT i = 0; i < k; ++i) {
            if (neq(in[idx1[i]], v1[i])) {
                std::cerr << "i=" << i << ", idx1[i]=" << idx1[i] << ", v1[i]=" << v1[i]
                          << ", but in[" << idx1[i] << "]=" << in[idx1[i]] << std::endl;
                throw CheckException("v1[]/idx1[] not match in[]");
            }
            if (neq(in[idx2[i]], v2[i])) {
                std::cerr << "i=" << i << ", idx2[i]=" << idx2[i] << ", v2[i]=" << v2[i]
                          << ", but in[" << idx2[i] << "]=" << in[idx2[i]] << std::endl;
                throw CheckException("v2[]/idx2[] not match in[]");
            }
        }

        std::vector<idxT> idx1_(idx1, idx1 + k);
        std::vector<idxT> idx2_(idx2, idx2 + k);
        std::sort(idx1_.begin(), idx1_.end());
        std::sort(idx2_.begin(), idx2_.end());
        if (std::unique(idx1_.begin(), idx1_.end()) != idx1_.end()) {
            throw CheckException("idx1 contains duplicated index");
        }
        if (std::unique(idx2_.begin(), idx2_.end()) != idx2_.end()) {
            throw CheckException("idx2 contains duplicated index");
        }
    }


    std::vector<T> v1_(v1, v1 + k);
    std::vector<T> v2_(v2, v2 + k);
    std::sort(v1_.begin(), v1_.end(), std::less<T>());
    std::sort(v2_.begin(), v2_.end(), std::less<T>());
    bool wrong = false;
    for (idxT i = 0; i < k; ++i) {
        if (neq(v1_[i], v2_[i])) {
            std::cerr << "i=" << i << ": " << v1_[i] << " != " << v2_[i] << std::endl;
            wrong = true;
            // break;
        }
    }
    if (wrong) {
        throw CheckException("v1 != v2");
    }
}

template<typename T, typename idxT>
void check_result(const T* in,
                  int batch_size,
                  idxT len,
                  const T* v1,
                  const idxT* idx1,
                  const T* v2,
                  const idxT* idx2,
                  idxT k) {
    for (int i = 0; i < batch_size; ++i) {
        check_result(in + i * len,
                     len,
                     v1 + i * k,
                     idx1 ? (idx1 + i * k) : nullptr,
                     v2 + i * k,
                     idx2 ? (idx2 + i * k) : nullptr,
                     k);
    }
}


#endif
