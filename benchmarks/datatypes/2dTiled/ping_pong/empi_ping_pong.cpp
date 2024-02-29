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
struct empi_ping_pong : public empi_benchmark<T> {
    using base = empi_benchmark<T>;
    using base::base;
    using base::m_ctx;
    using base::m_message_group;

    void run(benchmark_args &args) {
        // Layout default parameters
        const auto rank = m_message_group->rank();
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
        for(auto i = 0; i < matrix_size / num_cols; i++) {
            for(auto j = 0; j < num_cols; j++) { data[i * num_cols + j] = 'a' + i; }
        }

        std::vector<T> res(tile_size);
        res.reserve(tile_size);

        // static constexpr int X = -1;
        // // Clang-Format off
        // int data_tiled[] = {
        //     /* tile 0, 0 */ 1,
        //     2,
        //     3,
        //     6,
        //     7,
        //     8,
        //     X,
        //     X,
        //     X,
        //     /* tile 0, 1 */ 4,
        //     5,
        //     X,
        //     9,
        //     10,
        //     X,
        //     X,
        //     X,
        //     X,
        // };
        // // Clang-Format on

        auto submatrix = twoDtiled::build_mdspan(data, size, num_cols, tile_row_size, tile_col_size, tile_to_send,
        times);
        // Kokkos::mdspan<int, Kokkos::dextents<std::size_t, 2>, empi::layouts::tiled_layout::tiled_layout_impl> matrix{
        //     data_tiled,
        //     typename empi::layouts::tiled_layout::tiled_layout_impl::template mapping<Kokkos::dextents<std::size_t, 2>>(
        //         Kokkos::dextents<std::size_t, 2>{2, 5}, 3, 3)};
        // // print matrix
        // for(auto i = 0; i < matrix.extent(0); i++) {
        //     for(auto j = 0; j < matrix.extent(1); j++) { std::cout << matrix(i, j) << " "; }
        //     std::cout << std::endl;
        // }
        // std::cout << std::endl;

        // auto submatrix = empi::layouts::submatrix_layout::build(matrix, tile_row_size, tile_col_size, tile_to_send);

        m_message_group->run([&](empi::MessageGroupHandler<T, empi::Tag{0}, empi::NOSIZE> &mgh) {
            auto&& ptr = twoDtiled::compact_view(submatrix, times, m_message_group);

            for(auto iter = 0; iter < iterations; iter++) {
                if(rank == 0) {
                    mgh.send(ptr.get(), 1, tile_size);
                } else {
                    mgh.recv(res.data(), 0, tile_size, status);
                }
            }
            times.stop(timings::mpi);

            if (rank == 1){
                auto matrix = Kokkos::mdspan(res.data(), Kokkos::dextents<std::size_t, 2>(tile_row_size,
                tile_col_size));
                // check matrix
                for(auto i = 0; i < tile_row_size; i++) {
                    for(auto j = 0; j < tile_col_size; j++) {
                        if(matrix(i, j) != static_cast<char>('a' + i + (tile_to_send / (num_cols / tile_col_size)) *
                        tile_row_size)) {
                            std::cerr << "Error: " << matrix(i, j) << " != " << static_cast<char>('a' + i +
                            (tile_to_send / (num_cols / tile_col_size)) * tile_row_size) << std::endl; std::abort();
                        }
                    }
                }
            }

        });
    }
};


int main(int argc, char **argv) {
    benchmark_manager<empi_ping_pong<char>> bench_app{argc, argv, EMPI_BENCHMARK_NAME};
    auto &parser = bench_app.get_parser();
    parser.add_argument("-c", "--num-cols").help("Number of columns [size x c]").scan<'i', size_t>().required();
    parser.add_argument("--sub-n").help("Number of submatrix rows").scan<'i', size_t>().required();
    parser.add_argument("--sub-m").help("Number of submatrix columns").scan<'i', size_t>().required();
    parser.add_argument("--tile").help("Tile to send").scan<'i', size_t>().default_value(1);
    bench_app.run_benchmark();

    return 0;
}
