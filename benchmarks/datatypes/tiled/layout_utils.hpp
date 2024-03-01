#pragma once

#include "../../include/benchmark.hpp"
#include "empi/empi.hpp"

namespace tiled {

template<typename T>
    requires empi::details::has_data<T>
static void pack(T &in, T &out, size_t size, size_t A, size_t B, size_t tiled_size, int rank, benchmark_timer &times) {
    times.start(timings::mpi); 
    auto base_datatype = empi::details::mpi_type<T>::get_type();
    if(rank == 0) {
        times.start(timings::compact); 
        int position = 0;
        for(int i = 0; i < size; i++) {
            if(i % B < A) { MPI_Pack(&in[i], 1, base_datatype, out.data(), tiled_size, &position, MPI_COMM_WORLD); }
        }
        assert(position == tiled_size && "Position must be equal to the size of the packed data");
        times.stop(timings::compact);
    }
}

template<typename T>
    requires empi::details::has_data<T>
static void unpack(
    T &in, T &out, size_t size, size_t A, size_t B, size_t tiled_size, int rank, benchmark_timer &times) {
    times.start(timings::unpack);
    for(int i = 0, position = 0; i < tiled_size; i++) {
        MPI_Unpack(
            out.data(), tiled_size, &position, &in[i], 1, empi::details::mpi_type<T>::get_type(), MPI_COMM_WORLD);
    }
    times.stop(timings::unpack);
    times.stop(timings::mpi);
}


template<typename T>
    requires empi::details::has_data<T>
static auto build_mdspan(T &data, size_t A, size_t B, size_t tiled_size, benchmark_timer &times) {
    times.start(timings::view);
    Kokkos::dextents<size_t, 1> ext(tiled_size);
    auto view = empi::layouts::block_layout::build(data, ext, A, B);
    times.stop(timings::view);
    return view;
}

template<typename T>
    requires empi::details::has_data<T>
static auto compact_view(T &data, auto view, benchmark_timer &times, std::unique_ptr<empi::MessageGroup> &mg) {
    mg->barrier();
    times.start(timings::mpi);
    times.start(timings::compact); 
    auto &&ptr = empi::layouts::block_layout::compact(view);
    times.stop(timings::compact);
    if(mg->rank() != 0) { times.reset(timings::compact); }
    return std::move(ptr);
}

} // namespace tiled