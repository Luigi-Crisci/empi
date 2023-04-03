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
  double t_start, t_end, t_start_inner;
  constexpr int SCALE = 1000000;
  long nBytes;
  empi::Context ctx(&argc, &argv);
  auto message_group = ctx.create_message_group(MPI_COMM_WORLD);

  // ------ PARAMETER SETUP -----------
  n = atoi(argv[1]);
  max_iter = atoi(argv[2]);

  double mpi_time = 0.0;
  
  

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

    auto view = empi::layouts::block_layout::build(
        myarr, empi::stdex::dextents<size_t, 1>(view_size), std::span(&A, 1),
        std::span(&B, 1));

    message_group->run([&](empi::MessageGroupHandler<type> &mgh) {
      // Warmup
      mgh.barrier();
      mgh.Allgather(view, view_size, recv, view_size);
      mgh.barrier();

      if (message_group->rank() == 0)
        t_start = MPI_Wtime();

      auto &&ptr = empi::layouts::block_layout::compact(view);
      for (auto iter = 0; iter < max_iter; iter++) {
        mgh.Allgather(ptr.get(), view_size, recv, view_size);
      }

      message_group->barrier();
      if (message_group->rank() == 0) {
        t_end = MPI_Wtime();
        mpi_time = (t_end - t_start) * SCALE;
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