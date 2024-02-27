#ifndef C8F2C779_A596_4C43_87EA_99790575966A
#define C8F2C779_A596_4C43_87EA_99790575966A

#include "../../include/benchmark.hpp"
#include "empi/empi.hpp"

namespace twoDtiled {

template<typename T>
    requires empi::details::has_data<T>
static void pack(T &in, T &out, size_t row_num, size_t col_num, size_t tile_row_size, size_t tile_col_size, size_t tile_to_send, int rank, benchmark_timer &times) {
    auto base_datatype = empi::details::mpi_type<T>::get_type();
    if(rank == 0) {
        times.mpi_time[benchmark_timer::start] = times.compact_time[benchmark_timer::start] = empi::wtime();
        int position = 0;
        const size_t row_tile_pos = (tile_to_send / (col_num / tile_col_size)) * tile_row_size;
        const size_t col_tile_pos = (tile_to_send % (col_num / tile_col_size)) * tile_col_size;
        for(auto i = 0; i < tile_row_size; i++) {
            MPI_Pack(&in[row_tile_pos + i][col_tile_pos], tile_col_size, base_datatype, out.data() + position, tile_col_size * tile_row_size, &position, MPI_COMM_WORLD);
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
    // for(int i = 0, position = 0; i < tiled_size; i++) {




        
    //         out.data(), tiled_size, &position, &in[i], 1, empi::details::mpi_type<T>::get_type(), MPI_COMM_WORLD);
    // }
    times.mpi_time[benchmark_timer::end] = times.unpack_time[benchmark_timer::end] = MPI_Wtime();
}


template<typename T>
    requires empi::details::has_data<T>
static auto build_mdspan(T& data, size_t A, size_t B, size_t tiled_size, benchmark_timer &times){
    times.view_time[benchmark_timer::start] = empi::wtime();
    Kokkos::dextents<size_t, 1> ext(tiled_size);
    auto view = empi::layouts::block_layout::build(data, ext, A, B);
    times.view_time[benchmark_timer::end] = empi::wtime();
    return view;
}

template<typename T>
    requires empi::details::has_data<T>
static auto compact_view(T& data, auto view, benchmark_timer &times, std::unique_ptr<empi::MessageGroup>& mg){
    mg->barrier();
    times.mpi_time[benchmark_timer::start] = times.compact_time[benchmark_timer::start] = empi::wtime();
            auto&& ptr = empi::layouts::block_layout::compact(view); 
            times.compact_time[benchmark_timer::end] = empi::wtime();
            if (mg->rank() != 0) {
                 times.compact_time[benchmark_timer::start] = times.compact_time[benchmark_timer::end] = 0;
            }
            return ptr;
}

} // namespace tiled


#endif /* C8F2C779_A596_4C43_87EA_99790575966A */
