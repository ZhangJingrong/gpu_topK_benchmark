#include <unistd.h>

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "cpu_topk.h"
#include "factory.h"
#include "nv_util.h"
#include "test_util.h"

using namespace std;

using data_t = float;
using idx_t = int;


vector<string> split(const string& s, char delimiter) {
    vector<string> tokens;
    string token;
    istringstream iss(s);
    while (getline(iss, token, delimiter)) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    return tokens;
}

pair<float, float> calc_mean_and_stddev(const vector<float>& data) {
    double mean = 0.0;
    for (auto x : data) {
        mean += x;
    }
    mean /= data.size();

    double var = 0.0;
    for (auto x : data) {
        double diff = x - mean;
        var += diff * diff;
    }
    var /= data.size();

    return {mean, std::sqrt(var)};
}

void check_algos(const Factory<data_t, idx_t>& factory, const vector<string>& algos) {
    bool has_wrong_algo = false;
    for (const auto& algo : algos) {
        if (algo == "cpu") {
            continue;
        }
        if (!factory.has_algo(algo)) {
            cerr << "error: unknown algo: " << algo << endl;
            has_wrong_algo = true;
        }
    }
    if (has_wrong_algo) {
        exit(-1);
    }
}

vector<float> run_algo(const Factory<data_t, idx_t>& factory,
                       const string& algo,
                       int niter,
                       int warmup_niter,
                       bool do_check,
                       bool gaussian_distributed,
                       int num_random_bits,
                       int batch_size,
                       idx_t len,
                       idx_t k,
                       bool greater,
                       cudaStream_t stream) {
    Factory<data_t, idx_t>::topk_func_t topk_func = factory.create(algo);

    void* d_buf = nullptr;
    size_t buf_size;
    topk_func(nullptr,
              buf_size,
              nullptr,
              batch_size,
              len,
              k,
              nullptr,
              nullptr,
              greater,
              stream);
    assert(buf_size);
    TOPK_CUDA_CHECK(cudaMalloc((void**)&d_buf, buf_size));

    for (int i = 0; i < warmup_niter; ++i) {
        TestData<data_t, idx_t> data(
            batch_size, len, k, gaussian_distributed, num_random_bits);
        topk_func(d_buf,
                  buf_size,
                  data.d_in,
                  batch_size,
                  len,
                  k,
                  data.d_out,
                  data.d_out_idx,
                  greater,
                  stream);
        TOPK_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    vector<float> used_time;
    for (int i = 0; i < niter; ++i) {
        TestData<data_t, idx_t> data(
            batch_size, len, k, gaussian_distributed, num_random_bits);

        nv::Timer timer;
        topk_func(d_buf,
                  buf_size,
                  data.d_in,
                  batch_size,
                  len,
                  k,
                  data.d_out,
                  data.d_out_idx,
                  greater,
                  stream);
        TOPK_CUDA_CHECK(cudaStreamSynchronize(stream));
        used_time.push_back(timer.elapsed_ms());
        TOPK_CUDA_CHECK_LAST_ERROR();

        // to save time, only check correctness for the last iteration
        if (do_check && i == niter - 1) {
            data.d2h_in();
            data.d2h_out();

            nv::cpu_topk(data.h_in,
                         batch_size,
                         len,
                         k,
                         data.h_out_truth,
                         data.h_out_idx_truth,
                         greater);
            if (algo == "drtopk_bitonic") {
                check_result(data.h_in,
                             batch_size,
                             len,
                             data.h_out,
                             (const idx_t*)nullptr,
                             data.h_out_truth,
                             (const idx_t*)nullptr,
                             k);
            }
            else {
                check_result(data.h_in,
                             batch_size,
                             len,
                             data.h_out,
                             data.h_out_idx,
                             data.h_out_truth,
                             data.h_out_idx_truth,
                             k);
            }
        }
    }
    TOPK_CUDA_CHECK(cudaFree(d_buf));
    return used_time;
}


int main(int argc, char** argv) {
    int opt;
    bool do_check = false;
    int niter = 1;
    int warmup_niter = 1;
    int num_random_bits = sizeof(data_t) * 8;
    bool gaussian_distributed = 0;

    while ((opt = getopt(argc, argv, "cn:w:r:g")) != -1) {
        switch (opt) {
        case 'c':
            do_check = true;
            break;
        case 'n':
            niter = atoi(optarg);
            break;
        case 'w':
            warmup_niter = atoi(optarg);
            break;
        case 'r':
            num_random_bits = atoi(optarg);
            break;
        case 'g':
            gaussian_distributed = 1;
            break;
        default:
            cerr << "error: unknown option" << endl;
            return -1;
        }
    }
    if (niter < 1) {
        cerr << "-n <niter> should >= 1" << endl;
        return -1;
    }
    if (warmup_niter < 0) {
        cerr << "-w <warmup_niter> should >= 0" << endl;
        return -1;
    }
    if (num_random_bits != sizeof(data_t) * 8 && sizeof(data_t) != sizeof(unsigned)) {
        cerr << "-r should be use with " << sizeof(unsigned) * 8 << "-bits type" << endl;
        return -1;
    }
    if (num_random_bits != sizeof(data_t) * 8 && gaussian_distributed) {
        cerr << "only one of -r and -g can be specified" << endl;
        return -1;
    }

    Factory<data_t, idx_t> factory;
    if (argc - optind != 4) {
        cerr << "usage: " << argv[0]
             << " [-c] [-n niter] [-w warmup_niter] [-r num_random_bits] [-g] algos "
                "batch_size len k\n"
             << "   -c: compare with cpu result for validation\n"
             << "   -n: number of iterations for benchmark, default = 1\n"
             << "   -w: number of iterations for warmup, default = 1\n"
             << "   -r: number of random bits, can only be used for 32-bits type and "
                "can't be used with -g\n"
             << "   -g: use Gaussian distribution data (mean=0, stddev=1.0)\n"
             << "       meaningful only for float\n"
             << "       with this option, len should be multiple of 2 for float\n"
             << "\n"
             << "default distribution for test data:\n"
             << "       Uniform distribution in (0,1] for float\n"
             << "       bit-wise randomness for other data types\n"
             << "\n"
             << "algos: comma separated algorithm names, e.g. cpu,cub\n"
             << "available algorithms:\n"
             << "    cpu\n";
        vector<string> algo_names = factory.algo_names();
        for (const auto& algo : algo_names) {
            cerr << "    " << algo << endl;
        }
        return -1;
    }

    vector<string> algos = split(argv[optind++], ',');
    // use atof instead of atol so it supports argument like 1e5
    const int batch_size = atof(argv[optind++]);
    const idx_t len = atof(argv[optind++]);
    const idx_t k = atof(argv[optind++]);

    check_algos(factory, algos);
    if (k >= len) {
        cerr << "error: k must < len, but len=" << len << ", k=" << k << endl;
        return -1;
    }
    if (gaussian_distributed) {
        if (len % 2) {
            cerr << "error: with -g, len has to be multiple of 2 for float" << endl;
            return -1;
        }
    }

    cudaStream_t stream;
    TOPK_CUDA_CHECK(cudaStreamCreate(&stream));
    /*printf("%30s %10s %12s %10s %10s %10s\n",
           "algo",
           "batch_size",
           "len",
           "k",
           "time(ms)",
           "stddev");*/
    for (const auto& algo : algos) {
        // sample-select supports only greater=false
        const bool greater =
            (algo == "drtopk_radix" || algo == "drtopk_bitonic") ? true : false;
        vector<float> used_time;
        if (algo == "cpu") {
            for (int i = 0; i < niter; ++i) {
                TestData<data_t, idx_t> data(
                    batch_size, len, k, gaussian_distributed, num_random_bits);
                data.d2h_in();

                nv::Timer timer;
                nv::cpu_topk(
                    data.h_in, batch_size, len, k, data.h_out, data.h_out_idx, greater);
                used_time.push_back(timer.elapsed_ms());
            }
        }
        else {
            used_time = run_algo(factory,
                                 algo,
                                 niter,
                                 warmup_niter,
                                 do_check,
                                 gaussian_distributed,
                                 num_random_bits,
                                 batch_size,
                                 len,
                                 k,
                                 greater,
                                 stream);
        }
        pair<float, float> mean_and_stddev = calc_mean_and_stddev(used_time);
        printf("%10.3f, ", mean_and_stddev.first);
        /*printf("%30s %10d %12lld %10lld %10.3f %10.3f",
               algo.c_str(),
               batch_size,
               (long long int)len,
               (long long int)k,
               mean_and_stddev.first,
               mean_and_stddev.second);
        cout << endl;*/
    }
    TOPK_CUDA_CHECK(cudaStreamDestroy(stream));
}
