#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <malloc.h>
#include <mpi.h>
#include <mpl/mpl.hpp>
#include <unistd.h>
#include <vector>

using namespace std;
using value_type = char;


int main(int argc, char **argv) {
    int err, n, myid;
    double t_start, t_end, mpi_time = 0.0;
    constexpr int SCALE = 1000000;
    long pow_2_bytes, max_iter;

    MPI_Status status;

    // ------ PARAMETER SETUP -----------
    pow_2_bytes = strtol(argv[1], nullptr, 10);
    n = static_cast<int>(std::pow(2, pow_2_bytes));
    max_iter = strtol(argv[2], nullptr, 10);

    std::vector<value_type> arr(n);

    const mpl::communicator &comm_world(mpl::environment::comm_world());
    mpl::contiguous_layout<value_type> l(n);
    mpl::irequest_pool events;

    // Warmup
    comm_world.barrier();
    comm_world.bcast(0, arr.data(), l);
    comm_world.barrier();

    // Main Measurement
    if(comm_world.rank() == 0) t_start = mpl::environment::wtime();

    for(auto iter = 0; iter < max_iter; iter++) { comm_world.bcast(0, arr.data(), l); }

    comm_world.barrier();
    if(comm_world.rank() == 0) {
        t_end = mpl::environment::wtime();
        mpi_time = (t_end - t_start) * SCALE;
    }

    comm_world.barrier();

    if(comm_world.rank() == 0) {
        // cout << "\nData Size: " << nBytes << " bytes\n";
        cout << mpi_time << "\n";
        // cout << "Mean of communication times: " << Mean(mpi_time, num_restart)
        //      << "\n";
        // cout << "Median of communication times: " << Median(mpi_time,
        // num_restart)
        //      << "\n";
        // 	Print_times(mpi_time, num_restart);
    }
    return 0;
} // end main
