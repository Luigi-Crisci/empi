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
  int myid, procs, n, err, max_iter, sleep_time, iter = 0, range = 100, pow_2;
  double t_start, t_end, t_start_inner, t_compact1, t_compact2, t_view1, t_view2, compact_time;
  double mpi_time = 0.0;
  constexpr int SCALE = 1000000;
  constexpr int WARMUP = 100; 
  long nBytes;
  
  empi::Context ctx(&argc, &argv);
  auto message_group = ctx.create_message_group(MPI_COMM_WORLD);

  // ------ PARAMETER SETUP -----------
  n = atoi(argv[1]);
  max_iter = atoi(argv[2]);

  size_t A = std::stoi(argv[3]);
  size_t B = std::stoi(argv[4]);
  assert(B >= A);
  string datatype = argv[5];

  auto run_bench = [&](auto data_t_v) {
    using type = decltype(data_t_v);
    n = n / sizeof(type);
    assert(n > 0);

    size_t view_size = n / B * A;

    std::vector<type> myarr(n);
    std::vector<type> recv(view_size * message_group->size());

    t_view1 = MPI_Wtime();
    auto view = empi::layouts::block_layout::build(
        myarr, Kokkos::dextents<size_t, 1>(view_size), A,
        B);
    t_view2 = MPI_Wtime();

    message_group->run([&](empi::MessageGroupHandler<type> &mgh) {
      // Warmup
      mgh.barrier();
      for (auto iter = 0; iter < WARMUP; iter++)
        mgh.Allgather(view, view_size, recv, view_size);
      mgh.barrier();

      t_start = MPI_Wtime();
      auto &&ptr = empi::layouts::block_layout::compact(view);
      t_compact2 = MPI_Wtime();

      for (auto iter = 0; iter < max_iter; iter++) {
        mgh.Allgather(ptr.get(), view_size, recv, view_size);
      }

      message_group->barrier();
      t_end = MPI_Wtime();
      if (message_group->rank() == 0) {
        mpi_time = (t_end - t_compact2) * SCALE;
        compact_time = (t_compact2 - t_start) * SCALE;
      }
    });
  };

  if (datatype == "basic") {
    run_bench(char());
  } else {
    run_bench(basic_struct{});
  }

  message_group->barrier();
  if (message_group->rank() == 0) {
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

