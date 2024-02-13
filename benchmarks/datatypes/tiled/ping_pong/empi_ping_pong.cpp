#include <cstdio>
#include <ctime>
#include <empi/empi.hpp>
#include <malloc.h>
#include <mpi.h>
#include <unistd.h>

#include <argparse/argparse.hpp>

#include "../../../include/benchmark.hpp"
#include "../../../include/bench_templates.hpp"

using namespace std;

template<typename T>
struct empi_ping_pong : public empi_benchmark<T>{
    using base = empi_benchmark<T>;
    using base::base;
    using base::m_ctx;
    using base::m_message_group;

    void run(benchmark_args& args){
        const auto rank = m_message_group->rank();
        const size_t size = args.size;
        const size_t iterations = args.iterations;
        const size_t warmup_runs = args.warmup_runs;
        auto &times = args.times;
        MPI_Status status;
        size_t A = args.parser.get<size_t>("A");
        size_t B = args.parser.get<size_t>("B");
        auto tiled_size = size / B * A;

        std::vector<T> data(size);
        std::iota(data.begin(), data.end(), 0);
        std::vector<T> res(tiled_size);
        res.reserve(tiled_size);

        times.view_time[benchmark_timer::start] = empi::wtime();
        Kokkos::dextents<size_t, 1> ext(tiled_size);
        auto view = empi::layouts::block_layout::build(data, ext, A, B);
        times.view_time[benchmark_timer::end] = empi::wtime();


        m_message_group->run([&](empi::MessageGroupHandler<T, empi::Tag{0}, empi::NOSIZE> &mgh) {
            // Warm up
            mgh.barrier();

            for(auto iter = 0; iter < warmup_runs; iter++) {
                if(rank == 0) {
                    mgh.send(view, 1, tiled_size);
                    mgh.recv(res, 1, tiled_size, status);
                } else {
                    mgh.recv(view, 0, tiled_size, status);
                    mgh.send(res, 0, tiled_size);
                }
            }
            mgh.barrier();

            times.mpi_time[benchmark_timer::start] = times.compact_time[benchmark_timer::start] = empi::wtime();
            auto ptr = empi::layouts::block_layout::compact(view); 
            times.compact_time[benchmark_timer::end] = empi::wtime();

            for(auto iter = 0; iter < iterations; iter++) {
                if(rank == 0) {
                    mgh.send(ptr.get(), 1, tiled_size);
                    mgh.recv(res, 1, tiled_size, status);
                } else {
                    mgh.recv(res, 0, tiled_size, status);
                    mgh.send(res, 0, tiled_size);
                }
            }

            m_message_group->barrier();
        });
    }
};


int main(int argc, char **argv) {
    benchmark_manager<empi_ping_pong<char>> bench_app{argc, argv, EMPI_BENCHMARK_NAME};
    auto &parser = bench_app.get_parser();
    parser.add_argument("-A").help("Stride A").scan<'i', size_t>().required();
    parser.add_argument("-B").help("Stride B (B > A)").scan<'i', size_t>().required();

    bench_app.run_benchmark();

    return 0;
}

