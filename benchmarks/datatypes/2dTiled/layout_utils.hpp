#ifndef C8F2C779_A596_4C43_87EA_99790575966A
#define C8F2C779_A596_4C43_87EA_99790575966A

#include "../../include/benchmark.hpp"
#include "empi/empi.hpp"

#include <iostream>

namespace twoDtiled {

template<typename T>
    requires empi::details::has_data<T>
static void pack(T &in, T &out, size_t row_num, size_t col_num, size_t tile_row_size, size_t tile_col_size, size_t tile_to_send, int rank, benchmark_timer &times) {
    auto base_datatype = empi::details::mpi_type<T>::get_type();
    if(rank == 0) {
        times.mpi_time[benchmark_timer::start] = times.compact_time[benchmark_timer::start] = empi::wtime();
        int position = 0;
        // const size_t row_tile_pos = (tile_to_send / (col_num / tile_col_size)) * tile_row_size;
        const size_t col_tile_pos = (tile_to_send % (col_num / tile_col_size)) * tile_col_size;
        auto data = in.data();
        const auto tile_pos = tile_to_send * tile_row_size * tile_col_size;
        const auto tiles_per_row = (col_num / tile_col_size);
        const auto tile_row_pos = (tile_to_send / tiles_per_row) * tile_row_size;
        for(auto i = 0; i < tile_row_size; i++) {
            MPI_Pack(&data[tile_row_pos * col_num + (tile_to_send % tiles_per_row) * tile_col_size + i * col_num], tile_col_size, base_datatype, out.data(), tile_col_size * tile_row_size, &position, MPI_COMM_WORLD);
        }
        assert(position == tile_col_size * tile_row_size && "Position must be equal to the size of the packed data");
        times.compact_time[benchmark_timer::end] = MPI_Wtime();
    }
}

template<typename T>
    requires empi::details::has_data<T>
static void unpack(T &in, T &out, size_t row_num, size_t col_num, size_t tile_row_size, size_t tile_col_size, size_t tile_to_send, int rank, benchmark_timer &times) {
    times.mpi_time[benchmark_timer::end] = MPI_Wtime();
    times.unpack_time[benchmark_timer::start] = MPI_Wtime();
    // Unpack the data
    auto base_datatype = empi::details::mpi_type<T>::get_type();
    int position = 0;
    MPI_Unpack(in.data(), tile_col_size * tile_row_size, &position, out.data(), tile_col_size * tile_row_size, base_datatype, MPI_COMM_WORLD);
    assert(position == tile_col_size * tile_row_size && "Position must be equal to the size of the packed data");
    times.mpi_time[benchmark_timer::end] = times.unpack_time[benchmark_timer::end] = MPI_Wtime();
}


template<typename T>
    requires empi::details::has_data<T>
static auto build_mdspan(T& data, size_t size, size_t num_cols, size_t tile_row_size, size_t tile_col_size, size_t tile_to_send, benchmark_timer &times){
    times.view_time[benchmark_timer::start] = empi::wtime();
    auto matrix = Kokkos::mdspan(data.data(), Kokkos::dextents<std::size_t, 2>(size, num_cols));
    auto submatrix = empi::layouts::submatrix_layout::build(matrix, tile_row_size, tile_col_size, tile_to_send);      
    times.view_time[benchmark_timer::end] = empi::wtime();
    return submatrix;
}

template<typename T>
    requires empi::details::has_data<T>
static auto compact_view(T& data, auto view, benchmark_timer &times, std::unique_ptr<empi::MessageGroup>& mg){
    mg->barrier();
    times.mpi_time[benchmark_timer::start] = times.compact_time[benchmark_timer::start] = empi::wtime();
    auto&& ptr = empi::layouts::submatrix_layout::compact(view); 
    times.compact_time[benchmark_timer::end] = empi::wtime();
    if (mg->rank() != 0) {
            times.compact_time[benchmark_timer::start] = times.compact_time[benchmark_timer::end] = 0;
    }
    return std::move(ptr);


    

}

} // namespace two2tiled


#endif /* C8F2C779_A596_4C43_87EA_99790575966A */
