#include <algorithm>
#include <chrono>
#include <iostream>
#include <malloc.h>
#include <mpi.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#include "../../../include/utils.hpp"

using namespace std;


int main(int argc, char **argv) {
    int myid, procs, n, err, max_iter, nBytes, sleep_time, iter = 0, range = 100, pow_2;
    double t_start, t_end, t_datatype1, t_datatype2;
    constexpr int SCALE = 1000000;
    constexpr int WARMUP = 100;
    double mpi_time = 0.0;
    MPI_Status status;

    err = MPI_Init(&argc, &argv);
    if(err != MPI_SUCCESS) {
        cout << "\nError initializing MPI.\n";
        MPI_Abort(MPI_COMM_WORLD, err);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // ------ PARAMETER SETUP -----------
    n = atoi(argv[1]);
    max_iter = atoi(argv[2]);

    // Getting layout values
    int A1 = std::stoi(argv[3]);
    int A2 = std::stoi(argv[4]);
    int B = std::stoi(argv[5]);
    string datatype = argv[6];
    MPI_Datatype tiled_datatype;
    int flags;

    void *myarr, *arr;
    if(datatype == "basic") {
        myarr = allocate<char>(n / sizeof(char));
        arr = allocate<char>(n / sizeof(char));
    } else {
        myarr = allocate<basic_struct>(n / sizeof(basic_struct));
        arr = allocate<basic_struct>(n / sizeof(basic_struct));
    }

    t_datatype1 = MPI_Wtime();
    auto raw_datatype = get_datatype(datatype);
    bl_bucket(&tiled_datatype, &flags, raw_datatype, A1, A2, B);
    t_datatype2 = MPI_Wtime();

    int tiled_size = get_communication_size(n, tiled_datatype, raw_datatype);


    // if (myid == 0) {
    //   std::cout << "tiled size: " << tiled_size << "\n";
    //   std::cout << "Extent: " << extent << "\n";
    //   std::cout << "Size: " << size << "\n";
    //   std::cout << "Total size: " << tiled_size * size << "\n";
    //   std::cout << "Required memory: " << tiled_size * extent << "\n";
    // }

    // Warmup
    MPI_Barrier(MPI_COMM_WORLD);
    for(auto iter = 0; iter < WARMUP; iter++) {
        if(myid == 0) {
            MPI_Send(arr, tiled_size, tiled_datatype, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(arr, tiled_size, tiled_datatype, 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        } else { // Node rank 1
            MPI_Recv(myarr, tiled_size, tiled_datatype, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Send(myarr, tiled_size, tiled_datatype, 0, 1, MPI_COMM_WORLD);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    t_start = MPI_Wtime();
    for(auto iter = 0; iter < max_iter; iter++) {
        if(myid == 0) {
            MPI_Send(arr, tiled_size, tiled_datatype, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(arr, tiled_size, tiled_datatype, 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        } else { // Node rank 1
            MPI_Recv(myarr, tiled_size, tiled_datatype, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Send(myarr, tiled_size, tiled_datatype, 0, 1, MPI_COMM_WORLD);
        }
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
        // cout << "Median of communication times: " << Median(mpi_time ,
        // num_restart )
        //      << "\n";
        // Print_times(mpi_time, num_restart);
    }

    MPI_Finalize();

    free(arr);
    free(myarr);

    return 0;
} // end main
