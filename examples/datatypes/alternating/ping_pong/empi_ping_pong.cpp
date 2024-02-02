#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <empi/empi.hpp>
#include <iostream>
#include <malloc.h>
#include <mpi.h>
#include <unistd.h>

#include "../../utils.hpp"

using namespace std;


int main(int argc, char **argv) {
    int myid, procs, n, err, max_iter, nBytes, sleep_time, iter = 0, range = 100, pow_2;
    double t_start, t_end, t_start_inner, t_compact1, t_compact2, t_view1, t_view2, compact_time;
    double mpi_time = 0.0;
    constexpr int SCALE = 1000000;
    constexpr int WARMUP = 100;

    empi::Context ctx(&argc, &argv);

    // ------ PARAMETER SETUP -----------
    n = atoi(argv[1]);
    max_iter = atoi(argv[2]);

    auto message_group = ctx.create_message_group(MPI_COMM_WORLD);
    MPI_Status status;
    const int rank = message_group->rank();

    // Constucting layout
    size_t A1 = std::stoi(argv[3]);
    size_t A2 = std::stoi(argv[4]);
    size_t B1 = std::stoi(argv[5]);
    size_t B2 = std::stoi(argv[6]);
    string datatype = argv[7];

    auto run_bench = [&](auto data_t_v) {
        using type = decltype(data_t_v);
        n = n / sizeof(type);
        assert(n > 0);

        assert(A1 <= B1 && A2 <= B2);
        assert(n % (B1 + B2) == 0);
        auto num_blocks = (n / (B1 + B2)) * 2;
        auto half_block = num_blocks / 2;
        auto tiled_size = half_block * (A1 + A2);
        std::array sizes{A1, A2};
        std::array strides{B1, B2};

        std::vector<type> myarr(n);

        t_view1 = MPI_Wtime();
        Kokkos::dextents<size_t, 1> ext(tiled_size);
        auto view = empi::layouts::block_layout::build(myarr, ext, sizes, strides);
        t_view2 = MPI_Wtime();

        std::vector<type> res(tiled_size);

        message_group->run([&](empi::MessageGroupHandler<type, empi::Tag{0}, empi::NOSIZE> &mgh) {
            // Warm up
            mgh.barrier();
            for(auto iter = 0; iter < WARMUP; iter++) {
                if(rank == 0) {
                    mgh.send(view, 1, tiled_size);
                    mgh.recv(res, 1, tiled_size, status);
                } else {
                    mgh.recv(res, 0, tiled_size, status);
                    mgh.send(res, 0, tiled_size);
                }
            }
            mgh.barrier();

            t_start = MPI_Wtime();
            auto &&ptr = empi::layouts::block_layout::compact(view);
            t_compact2 = MPI_Wtime();

            for(auto iter = 0; iter < max_iter; iter++) {
                if(rank == 0) {
                    mgh.send(ptr.get(), 1, tiled_size);
                    mgh.recv(res, 1, tiled_size, status);
                } else {
                    mgh.recv(res, 0, tiled_size, status);
                    mgh.send(res, 0, tiled_size);
                }
            }

            message_group->barrier();

            t_end = MPI_Wtime();
            if(message_group->rank() == 0) {
                mpi_time = (t_end - t_compact2) * SCALE;
                compact_time = (t_compact2 - t_start) * SCALE;
            }
        });
    };

    if(datatype == "basic") {
        run_bench(char());
    } else {
        run_bench(basic_struct{});
    }

    message_group->barrier();

    if(message_group->rank() == 0) {
        // cout << "\nData Size: " << nBytes << " bytes\n";
        cout << mpi_time << "\n";
        cout << ((t_view2 - t_view1) * SCALE) << "\n";
        cout << compact_time << "\n";
        // cout << "Mean of communication times: " << Mean(mpi_time, num_restart)
        //      << "\n";
        // cout << "Median of communication times: " << Median(mpi_time,
        // num_restart)
        //      << "\n";
        // 	Print_times(mpi_time, num_restart);
    }
    return 0;
} // end main
