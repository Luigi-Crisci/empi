#include <chrono>
#include <cmath>
#include <iostream>
#include <malloc.h>
#include <mpi.h>
#include <cstdio>
#include <ctime>
#include <unistd.h>
#include <empi/empi.hpp>

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

  // ------ PARAMETER SETUP -----------
  pow_2 = atoi(argv[1]);
  max_iter = atoi(argv[2]);
 
  double mpi_time = 0.0;
  nBytes = std::pow(2, pow_2);
  n = nBytes;
  auto message_group = ctx.create_message_group(MPI_COMM_WORLD);

  std::vector<char> myarr(n,'0');
  // Constucting layout
  size_t A1 = std::stoi(argv[3]);
  size_t A2 = std::stoi(argv[4]);
  size_t B  = std::stoi(argv[5]);
  
  assert( n % B == 0);
  auto num_blocks = n / B;
  auto half_block = num_blocks / 2;
  auto tiled_size = half_block * (A1+A2);
  empi::stdex::dextents<size_t, 1> ext(tiled_size);
  std::array sizes{A1,A2};
  std::array strides{B,B};

  auto view = empi::layouts::block_layout::build(myarr, 
                                                ext,
                                                sizes,
                                                strides);

      message_group->run([&](empi::MessageGroupHandler<char, empi::Tag{0}, empi::NOSIZE> &mgh) {
          // Warmup
          mgh.barrier();
          mgh.Bcast(view,0,tiled_size);
          mgh.barrier();

          if (message_group->rank() == 0)
              t_start = MPI_Wtime();

          auto&& ptr = empi::layouts::block_layout::compact(view);
          for (auto iter = 0; iter < max_iter; iter++) {
            mgh.Bcast(ptr.get(),0,tiled_size);            
          }

          message_group->barrier();
          if (message_group->rank() == 0) {
            t_end = MPI_Wtime();
            mpi_time =
                (t_end - t_start) * SCALE;
          }
      });

      
  message_group->barrier();
  if (message_group->rank() == 0) {
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