#include <chrono>
#include <cmath>
#include <iostream>
#include <malloc.h>
#include <mpi.h>
#include <cstdio>
#include <ctime>
#include <unistd.h>
#include <mpl/mpl.hpp>
#include <vector>

using namespace std;
using value_type = char;



int main(int argc, char **argv) {
  double t_start, t_end;
  double mpi_time = 0.0;
  constexpr int SCALE = 1000000;
  int err;
  long pow_2_bytes;
  int n;
  int myid;
  long max_iter;

  MPI_Status status;

  pow_2_bytes = strtol(argv[1], nullptr, 10);
  n = static_cast<int>(std::pow(2, pow_2_bytes));
  max_iter = strtol(argv[2], nullptr, 10);

  std::vector<value_type> arr(n);

  const mpl::communicator &comm_world(mpl::environment::comm_world());
  mpl::contiguous_layout<value_type> l(n);

   int _succ, _prev;
    _succ = (comm_world.rank() + 1) % comm_world.size();
    _prev = comm_world.rank() == 0 ? (comm_world.size() - 1) : (comm_world.rank() - 1);

    mpl::irequest_pool events;

    // Warmup
    comm_world.barrier();
    auto r1{comm_world.isend(arr.data(),l, _prev)};
    auto r2{comm_world.isend(arr.data(),l, _succ)};
    auto r3{comm_world.irecv(arr.data(),l, _prev)};
    auto r4{comm_world.irecv(arr.data(),l, _succ)};

    r1.wait();
    r2.wait();
    r3.wait();
    r4.wait();
    comm_world.barrier();
    
    if (comm_world.rank() == 0)
        t_start = mpl::environment::wtime();

    for (auto iter = 0; iter < max_iter; iter++) {
        r1 = std::move(comm_world.isend(arr.data(), l, _prev));
        r2 = std::move(comm_world.isend(arr.data(), l, _succ));
        r3 = std::move(comm_world.irecv(arr.data(), l, _prev));
        r4 = std::move(comm_world.irecv(arr.data(), l, _succ));

        r1.wait();
        r2.wait();
        r3.wait();
        r4.wait();
      }
      
    comm_world.barrier();
    if (comm_world.rank() == 0) {
      t_end = mpl::environment::wtime();
      mpi_time =
          (t_end - t_start) * SCALE;
    }

  comm_world.barrier();

  if (comm_world.rank() == 0) {
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

