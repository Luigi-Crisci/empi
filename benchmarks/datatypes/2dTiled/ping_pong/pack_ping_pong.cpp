#include <cstdio>
#include <ctime>
#include <empi/empi.hpp>
#include <malloc.h>
#include <mpi.h>
#include <unistd.h>

#include <argparse/argparse.hpp>

#include "../../../include/bench_templates.hpp"
#include "../../../include/benchmark.hpp"
#include "../layout_utils.hpp"


using namespace std;

template<typename T>
struct pack_ping_pong : public mpi_benchmark<T> {
    using base = mpi_benchmark<T>;
    using base::base;

    void run(benchmark_args &args) {
        // Layout default parameters
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        const size_t size = args.size;
        const size_t iterations = args.iterations;
        auto &times = args.times;
        const size_t num_cols = args.parser.get<size_t>("--num-cols");
        const size_t matrix_size = size * num_cols;
        const size_t tile_row_size = args.parser.get<size_t>("--sub-n");
        const size_t tile_col_size = args.parser.get<size_t>("--sub-m");
        const size_t tile_size = tile_row_size * tile_col_size;
        const size_t num_tiles = matrix_size / tile_size;
        assert(matrix_size % tile_size == 0 && "Matrix size must be divisible by tile size");
        assert(num_cols % tile_col_size == 0 && "Number of columns must be divisible by tile column size");
        assert(size % tile_row_size == 0 && "Size must be divisible by tile row size");
        const size_t tile_to_send = args.parser.get<size_t>("--tile");
        assert(tile_to_send < num_tiles && "Tile to send must be less than number of tiles");

        MPI_Status status;

        std::vector<T> data(matrix_size);
        // Fill each tile with a different value
        for(auto i = 0; i < matrix_size /  num_cols; i++) {
            for(auto j = 0; j < num_cols; j++) {
                data[i * num_cols + j] = 'a' + i;
            }
        }
        
        std::vector<T> res(tile_size);
        std::vector<T> submatrix(tile_size);
        res.reserve(tile_size);

        MPI_Barrier(MPI_COMM_WORLD);

        twoDtiled::pack(data, submatrix, size, num_cols, tile_row_size, tile_col_size, tile_to_send, rank, times);

        for(auto iter = 0; iter < iterations; iter++) {
            if(rank == 0) {
                MPI_Send(submatrix.data(), tile_size, MPI_PACKED, 1, 0, MPI_COMM_WORLD);
            } else {
                MPI_Recv(res.data(), tile_size, MPI_PACKED, 0, 0, MPI_COMM_WORLD, &status);
            }
        }
        twoDtiled::unpack(res, submatrix, size, num_cols, tile_row_size, tile_col_size, tile_to_send, rank, times);

        if(rank == 1) {
            auto matrix = Kokkos::mdspan(submatrix.data(), Kokkos::dextents<std::size_t, 2>(tile_row_size, tile_col_size));
            // check matrix
           for(auto i = 0; i < tile_row_size; i++) {
                    for(auto j = 0; j < tile_col_size; j++) {
                        if(matrix(i, j) != static_cast<char>('a' + i + (tile_to_send / (num_cols / tile_col_size)) * tile_row_size)) {
                            std::cerr << "Error: " << matrix(i, j) << " != " << static_cast<char>('a' + i + (tile_to_send / (num_cols / tile_col_size)) * tile_row_size) << std::endl;
                            std::abort();
                        }
                    }
                }
        }
}
}
;


int main(int argc, char **argv) {
    benchmark_manager<pack_ping_pong<char>> bench_app{argc, argv, EMPI_BENCHMARK_NAME};
    auto &parser = bench_app.get_parser();
    parser.add_argument("-c", "--num-cols").help("Number of columns [size x c]").scan<'i', size_t>().required();
    parser.add_argument("--sub-n").help("Number of submatrix rows").scan<'i', size_t>().required();
    parser.add_argument("--sub-m").help("Number of submatrix columns").scan<'i', size_t>().required();
    parser.add_argument("--tile").help("Tile to send").scan<'i', size_t>().default_value(1);
    bench_app.run_benchmark();

    return 0;
}
