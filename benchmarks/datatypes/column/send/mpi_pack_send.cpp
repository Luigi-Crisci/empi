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
        constexpr size_t num_columns = 4;
        constexpr size_t col_to_send = 2;
        size_t view_size = num_columns * num_rows;
        auto &times = args.times;
        MPI_Status status;
        MPI_Datatype column_datatype;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        std::vector<T> data(view_size);      // 2 * N matrix matrix
        std::vector<T> rec_column(num_rows); // 2 * N matrix matrix
        // a a a a a
        // b b b b b ...
        if(rank == 0) {
            for(int i = 0; i < num_rows; i++) {
                for(int j = 0; j < num_columns; j++) { data[j + i * num_columns] = i % 10; }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        // Pack
        std::vector<T> send_col(num_rows);
        if(rank == 0) {
            times.mpi_time[benchmark_timer::start] = times.compact_time[benchmark_timer::start] = empi::wtime();
            for(int i = 0, position = 0; i < num_rows; i++) {
                MPI_Pack(&data[col_to_send + i * num_columns], 1, empi::details::mpi_type<T>::get_type(),
                    send_col.data(), num_rows * sizeof(T), &position, MPI_COMM_WORLD);
            }
            times.compact_time[benchmark_timer::end] = empi::wtime();
        }

        for(auto iter = 0; iter < iterations; iter++) {
            if(rank == 0) {
                MPI_Send(send_col.data(), num_rows, MPI_PACKED, 1, 0, MPI_COMM_WORLD);
            } else { // Node rank 1
                MPI_Recv(rec_column.data(), num_rows, MPI_PACKED, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            }
        }

        // Unpack
        if(rank == 1) {
            times.unpack_time[benchmark_timer::start] = empi::wtime();
            for(int i = 0, position = 0; i < num_rows; i++) {
                MPI_Unpack(rec_column.data(), num_rows * sizeof(T), &position, &rec_column[i], 1, empi::details::mpi_type<T>::get_type(),
                    MPI_COMM_WORLD);
            }
            times.mpi_time[benchmark_timer::end] = times.unpack_time[benchmark_timer::end] = MPI_Wtime();
        }
        MPI_Barrier(MPI_COMM_WORLD);

        if(rank == 1) {
            // Verify
            for(auto i = 0; i < num_rows; i++) {
                if(rec_column[i] != i % 10) {
                    std::cerr << "Error at index " << i << " value: " << rec_column[i] << std::endl;
                    std::exit(-1);
                }
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
