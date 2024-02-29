#include <cstdio>
#include <ctime>
#include <empi/empi.hpp>
#include <iostream>
#include <malloc.h>
#include <mpi.h>
#include <unistd.h>

#include "../../../include/bench_templates.hpp"
#include "../../../include/benchmark.hpp"

template<typename T>
struct empi_send : public empi_benchmark<T> {
    using base = empi_benchmark<T>;
    using base::base;
    using base::m_ctx;
    using base::m_message_group;

    void run(benchmark_args &args) {
        const auto rank = m_message_group->rank();
        const size_t num_rows = args.size;
        const size_t iterations = args.iterations;
        constexpr size_t num_columns = 4;
        constexpr size_t col_to_send = 2;
        size_t view_size = num_columns * num_rows;
        auto &times = args.times;
        MPI_Status status;

        std::vector<T> data(view_size);      // 2 * N matrix matrix
        std::vector<T> rec_column(num_rows); // 2 * N matrix matrix
        // a a a a a
        // b b b b b ...
        if(m_message_group->rank() == 0) {
            for(int i = 0; i < num_rows; i++) {
                for(int j = 0; j < num_columns; j++) { data[j + i * num_columns] = i % 10; }
            }
        }


        times.start(timings::view);
        Kokkos::extents<size_t, Kokkos::dynamic_extent, num_columns> extents(num_rows);
        auto view = empi::layouts::column_layout::build(data, extents, col_to_send);
        times.stop(timings::view);

        m_message_group->run([&](empi::MessageGroupHandler<T, empi::Tag{0}, empi::NOSIZE> &mgh) {
            mgh.barrier();
            
            times.start(timings::mpi); 
            times.start(timings::compact); 
            
            auto &&ptr = empi::layouts::compact(view); // basic compact function
            
            times.stop(timings::compact);
            if(rank != 0) {
                times.reset(timings::compact); // skip compact time for non root
            }
 
            for(auto iter = 0; iter < iterations; iter++) {
                if(rank == 0) {
                    mgh.send(ptr.get(), 1, num_rows);
                } else {
                    mgh.recv(rec_column, 0, num_rows, status);
                }
            }
            times.stop(timings::mpi);

            // Verify
            if(rank == 1) {
                for(auto i = 0; i < view.size(); i++) {
                    if(rec_column[i] != i % 10) {
                        std::cerr << "Error at index " << i << " value: " << view[i] << std::endl;
                        std::exit(-1);
                    }
                }
            }
        });
    }
};

int main(int argc, char **argv) {
    benchmark_manager<empi_send<char>> bench_app{argc, argv, EMPI_BENCHMARK_NAME};
    auto &parser = bench_app.get_parser();
    parser.add_argument("--num-columns").help("Number of columns in the matrix").scan<'i', size_t>().default_value(5);

    bench_app.run_benchmark();

    return 0;
}
