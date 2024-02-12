#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <malloc.h>
#include <mpi.h>
#include <unistd.h>

#include "../../../include/utils.hpp"

using namespace std;


int main(int argc, char **argv) {
    int myid, procs, err, max_iter, nBytes, sleep_time, range = 100, pow_2;
    double t_start, t_end, mpi_time = 0, t_datatype1, t_datatype2;
    long n;
    constexpr int SCALE = 1000000;
    constexpr int WARMUP = 100;

    err = MPI_Init(&argc, &argv);
    if(err != MPI_SUCCESS) {
        cout << "\nError initializing MPI\n";
        MPI_Abort(MPI_COMM_WORLD, err);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // ------ PARAMETER SETUP -----------
    n = std::stoi(argv[1]);
    max_iter = std::stoi(argv[2]);

    void *arr, *recv;
    // Getting layout values
    int A = std::stoi(argv[3]);
    int B = std::stoi(argv[4]);
    string datatype = argv[5];
    MPI_Datatype tiled_datatype;
    int flags;

    if(datatype == "basic") {
        arr = allocate<char>(n / sizeof(char));
        recv = allocate<char>(n * procs / sizeof(char));
    } else {
        arr = allocate<basic_struct>(n / sizeof(basic_struct));
        recv = allocate<basic_struct>(n * procs / sizeof(basic_struct));
    }

    t_datatype1 = MPI_Wtime();
    auto raw_datatype = get_datatype(datatype);
    bl_tiled(&tiled_datatype, &flags, A, B, raw_datatype);
    t_datatype2 = MPI_Wtime();

    int tiled_size = get_communication_size(n, tiled_datatype, raw_datatype);

    // Warmup
    MPI_Barrier(MPI_COMM_WORLD);
    for(auto iter = 0; iter < WARMUP; iter++)
        MPI_Allgather(arr, tiled_size, tiled_datatype, recv, tiled_size, tiled_datatype, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // main measurement
    t_start = MPI_Wtime();

    for(auto iter = 0; iter < max_iter; iter++) {
        MPI_Allgather(arr, tiled_size, tiled_datatype, recv, tiled_size, tiled_datatype, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t_end = MPI_Wtime();
    if(myid == 0) mpi_time = (t_end - t_start) * SCALE;

    MPI_Barrier(MPI_COMM_WORLD);

    if(myid == 0) {
        // cout << "\nData Size: " << nBytes << " bytes\n";
        cout << ((t_datatype2 - t_datatype1) * SCALE) << "\n";
        cout << mpi_time << "\n";
        // cout << "Mean of communication times: " << Mean(mpi_time , num_restart)
        //      << "\n";
        // cout << "Median of communication times: " << Median(mpi_time , num_restart )
        //      << "\n";
        // Print_times(mpi_time, num_restart);
    }
    free(arr);
    free(recv);
    MPI_Finalize();
    return 0;
} // end main
