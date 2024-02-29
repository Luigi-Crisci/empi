#include <iostream>
#include <malloc.h>
#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

#include "../../../include/bench_templates.hpp"
#include "../../../include/benchmark.hpp"
#include "../../../include/utils.hpp"
#include "../layout_utils.hpp"
#include "empi/datatype.hpp"

using namespace std;

template<typename T>
struct pack_bcast : public mpi_benchmark<T> {
    using base = mpi_benchmark<T>;
    using base::base;

    void run(benchmark_args &args) {
        const size_t size = args.size;
        const size_t iterations = args.iterations;
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        auto &times = args.times;
        size_t A = args.parser.get<size_t>("A");
        size_t B = args.parser.get<size_t>("B");
        assert(size % B == 0 && "Size must be divisible by B");
        auto tiled_size = size / B * A;

        std::vector<T> data(size);
        // a a a a a
        // b b b b b ...
        if(rank == 0) {
            for(size_t i = 0; i < size; i++) {
                if(i % B < A) { data[i] = 'a'; }
            }
        }

        std::vector<T> res(tiled_size);
        res.reserve(tiled_size);

        tiled::pack(data, res, size, A, B, tiled_size, rank, times);

        MPI_Barrier(MPI_COMM_WORLD);
        times.start(timings::mpi);
        for(auto iter = 0; iter < iterations; iter++) {
            MPI_Bcast(res.data(), tiled_size, MPI_PACKED, 0, MPI_COMM_WORLD);
        }

        tiled::unpack(data, res, size, A, B, tiled_size, rank, times);
        
        // Verify
        if(rank == 0) { data = res; }
        for(auto i = 0; i < tiled_size; i++) {
            if(data[i] != 'a') {
                std::cerr << "Error at index " << i << " value: " << data[i] << std::endl;
                std::abort();
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
};

int main(int argc, char **argv) {
    benchmark_manager<pack_bcast<char>> bench_app{argc, argv, EMPI_BENCHMARK_NAME};
    auto &parser = bench_app.get_parser();
    parser.add_argument("-A").help("Stride A").scan<'i', size_t>().required();
    parser.add_argument("-B").help("Stride B (B > A)").scan<'i', size_t>().required();

    bench_app.run_benchmark();

    return 0;
}
