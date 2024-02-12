#include <iostream>
#include <malloc.h>
#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

#include "../../../include/bench_templates.hpp"
#include "../../../include/benchmark.hpp"
#include "../../../include/utils.hpp"
#include "empi/datatype.hpp"

using namespace std;

template<typename T>
struct mpi_send : public mpi_benchmark<T> {
    using base = mpi_benchmark<T>;
    using base::base;

    void run(benchmark_args &args) {
        int rank;
        const size_t num_rows = args.size;
        const size_t iterations = args.iterations;
        const size_t warmup_runs = args.warmup_runs;
        constexpr size_t num_columns = 4;
        constexpr size_t col_to_send = 2;
        size_t view_size = num_columns * num_rows;
        auto &times = args.times;
        MPI_Datatype column_datatype;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        std::vector<T> data(view_size);      // 2 * N matrix matrix
        std::vector<T> rec_column(num_rows); // 2 * N matrix matrix
        // a a a a a
        // b b b b b ...
        if(rank == 0) {
            for(int i = 0; i < num_rows; i++) {
                for(int j = 0; j < num_columns; j++) { data[j + i * num_columns] = 'a' + i % 10; }
            }
        }
        times.view_time[benchmark_timer::start] = MPI_Wtime();
        auto raw_datatype = empi::details::mpi_type<T>::get_type();
        bl_column(&column_datatype, num_rows, num_columns, raw_datatype);
        times.view_time[benchmark_timer::end] = MPI_Wtime();
        int column_size = 1;


        // Warmup
        MPI_Barrier(MPI_COMM_WORLD);
        for(auto iter = 0; iter < warmup_runs; iter++) {
            MPI_Bcast(data.data() + col_to_send, column_size, column_datatype, 0, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        times.mpi_time[benchmark_timer::start] = MPI_Wtime();
        for(auto iter = 0; iter < iterations; iter++) {
            MPI_Bcast(data.data() + col_to_send, column_size, column_datatype, 0, MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        times.mpi_time[benchmark_timer::end] = MPI_Wtime();

        // Verify
        for(size_t i = col_to_send, j = 0; i < view_size; i+=num_columns, j++) {
            if(data[i] != 'a' + j % 10) {
                std::cerr << "Error at index " << i << " value: " << data[i] << std::endl;
                std::exit(-1);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
};

int main(int argc, char **argv) {
    benchmark_manager<mpi_send<char>> bench_app{argc, argv, EMPI_BENCHMARK_NAME};
    auto &parser = bench_app.get_parser();
    parser.add_argument("--num-columns").help("Number of columns in the matrix").scan<'i', size_t>().default_value(5);

    bench_app.run_benchmark();

    return 0;
}
