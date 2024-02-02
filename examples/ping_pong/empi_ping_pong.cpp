/*
 * Copyright (c) 2022-2023 University of Salerno, Italy. All rights reserved.
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <empi/empi.hpp>
#include <iostream>
#include <malloc.h>
#include <mpi.h>
#include <unistd.h>

using namespace std;



int main(int argc, char **argv) {
    int myid, procs, n, err, max_iter, nBytes, sleep_time, iter = 0, range = 100, pow_2;
    double t_start, t_end, t_start_inner;
    constexpr int SCALE = 1000000;

    empi::Context ctx(&argc, &argv);

    // ------ PARAMETER SETUP -----------
    pow_2 = atoi(argv[1]);
    max_iter = atoi(argv[2]);

    double mpi_time = 0.0;
    nBytes = std::pow(2, pow_2);
    n = nBytes;
    std::vector<char> myarr(n, 0);

    auto message_group = ctx.create_message_group(MPI_COMM_WORLD);
    MPI_Status status;
    const int rank = message_group->rank();

    message_group->run([&](empi::MessageGroupHandler<char, empi::Tag{0}, empi::NOSIZE> &mgh) {
        // Warm up
        mgh.barrier();
        if(rank == 0) {
            mgh.send(myarr.data(), 1, n);
            mgh.recv(myarr.data(), 1, n, status);
        } else {
            mgh.recv(myarr.data(), 0, n, status);
            mgh.send(myarr.data(), 0, n);
        }
        mgh.barrier();

        if(message_group->rank() == 0) t_start = MPI_Wtime();

        for(auto iter = 0; iter < max_iter; iter++) {
            if(rank == 0) {
                mgh.send(myarr.data(), 1, n);
                mgh.recv(myarr.data(), 1, n, status);
            } else {
                mgh.recv(myarr.data(), 0, n, status);
                mgh.send(myarr.data(), 0, n);
            }
        }

        message_group->barrier();
        if(message_group->rank() == 0) {
            t_end = MPI_Wtime();
            mpi_time = (t_end - t_start) * SCALE;
        }
    });

    message_group->barrier();

    if(message_group->rank() == 0) {
        // cout << "\nData Size: " << nBytes << " bytes\n";
        cout << mpi_time << "\n";
        // cout << "Mean of communication times: " << Mean(mpi_time, num_restart)
        //      << "\n";
        // cout << "Median of communication times: " << Median(mpi_time, num_restart)
        //      << "\n";
        // 	Print_times(mpi_time, num_restart);
    }
    return 0;
} // end main

