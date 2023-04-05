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

double Mean(double[], int);
double Median(double[], int);
void Print_times(double[], int);

int main(int argc, char **argv) {
  int myid, procs, n, err, max_iter, sleep_time, iter = 0, range = 100, pow_2;
  double t_start, t_end, t_start_inner, t_compact1, t_compact2, t_view1, t_view2, compact_time;
  double mpi_time = 0.0;
  constexpr int SCALE = 1000000;
  constexpr int WARMUP = 100;

  long nBytes;
  empi::Context ctx(&argc, &argv);

  // ------ PARAMETER SETUP -----------
  n = atoi(argv[1]);
  max_iter = atoi(argv[2]);
 
  auto message_group = ctx.create_message_group(MPI_COMM_WORLD);

  // Constucting layout
  size_t A = std::stoi(argv[3]);
  size_t B1 = std::stoi(argv[4]);
  size_t B2 = std::stoi(argv[5]);
  string datatype = argv[6];

  auto run_bench = [&](auto data_t_v) {
    using type = decltype(data_t_v);
    n = n / sizeof(type);
    assert(n > 0);

    assert(A <= B1 && A <= B2);
    assert(n % (B1 + B2) == 0);
    auto num_blocks = (n / (B1 + B2)) * 2;
    auto tiled_size = num_blocks * A;
    std::array sizes{A, A};
    std::array strides{B1, B2};

    std::vector<type> myarr(n);

    t_view1 = MPI_Wtime();
    empi::stdex::dextents<size_t, 1> ext(tiled_size);
    auto view = empi::layouts::block_layout::build(myarr, ext, sizes, strides);
    t_view2 = MPI_Wtime();

    message_group->run(
        [&](empi::MessageGroupHandler<type, empi::Tag{0}, empi::NOSIZE> &mgh) {
          // Warmup
          mgh.barrier();
          for (auto iter = 0; iter < WARMUP; iter++)
            mgh.Bcast(view, 0, tiled_size);
          mgh.barrier();

          t_start = MPI_Wtime();
          auto &&ptr = empi::layouts::block_layout::compact(view);
          t_compact2 = MPI_Wtime();

          for (auto iter = 0; iter < max_iter; iter++) {
            mgh.Bcast(ptr.get(), 0, tiled_size);
          }

          message_group->barrier();
          t_end = MPI_Wtime();
          if (message_group->rank() == 0) {
            mpi_time = (t_end - t_start) * SCALE;
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

double Mean(double a[], int n) {
  double sum = 0.0;
  for (int i = 0; i < n; i++)
    sum += a[i];

  return (sum / (double)n);
}

double Median(double a[], int n) {
  sort(a, a + n);
  if (n % 2 != 0)
    return a[n / 2];

  return (a[(n - 1) / 2] + a[n / 2]) / 2.0;
}

void Print_times(double a[], int n) {
  cout << "\n------------------------------------";
  for (int t = 0; t < n; t++)
    cout << "\n " << a[t];
}