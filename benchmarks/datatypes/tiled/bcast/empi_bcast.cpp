#include <cstdio>
#include <ctime>
#include <empi/empi.hpp>
#include <malloc.h>
#include <mpi.h>
#include <unistd.h>

#include <argparse/argparse.hpp>

#include "../../../include/benchmark.hpp"
#include "../../../include/bench_templates.hpp"
#include "../layout_utils.hpp"

using namespace std;

template<typename T>
struct empi_bcast : public empi_benchmark<T>{
    using base = empi_benchmark<T>;
    using base::base;
    using base::m_ctx;
    using base::m_message_group;

    void run(benchmark_args& args){
        const auto rank = m_message_group->rank();
        const size_t size = args.size;
        const size_t iterations = args.iterations;
        auto &times = args.times;
        MPI_Status status;
        size_t A = args.parser.get<size_t>("A");
        size_t B = args.parser.get<size_t>("B");
        assert(size % B == 0 && "Size must be divisible by B");
        auto tiled_size = size / B * A;

        std::vector<T> data(size);
        // a a a a a
        // b b b b b ...
        if(m_message_group->rank() == 0) {
           for(size_t i = 0; i < size; i++) {
               if (i % B < A){
                data[i] = 'a';
               }
           }
        }

        std::vector<T> res(tiled_size);
        res.reserve(tiled_size);

        auto view = tiled::build_mdspan(data, A, B, tiled_size, times);

        m_message_group->run([&](empi::MessageGroupHandler<T, empi::Tag{0}, empi::NOSIZE> &mgh) {
            auto&& ptr = tiled::compact_view(data, view, times, m_message_group);

            for(auto iter = 0; iter < iterations; iter++) {
                mgh.Bcast(ptr.get(), 0, tiled_size);
            }
    
            times.stop(timings::mpi);

            for(auto i = 0; i < res.size(); i++) {
                if(ptr.get()[i] != 'a') {
                    std::cerr << "Error: " << res[i] << " != " << 'a' << std::endl;
                    std::abort();
                }
            }
        }); 
    }
};


int main(int argc, char **argv) {
    benchmark_manager<empi_bcast<char>> bench_app{argc, argv, EMPI_BENCHMARK_NAME};
    auto &parser = bench_app.get_parser();
    parser.add_argument("-A").help("Stride A").scan<'i', size_t>().required();
    parser.add_argument("-B").help("Stride B (B > A)").scan<'i', size_t>().required();

    bench_app.run_benchmark();

    return 0;
}

