#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <malloc.h>
#include <mpi.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

using namespace std;



int main(int argc, char **argv) {
  int myid, procs, err, max_iter, nBytes, sleep_time, range = 100, pow_2;
  double t_start, t_end, mpi_time = 0;
  long n;
  constexpr int SCALE = 1000000;
  char *arr;

  MPI_Status status;
  err = MPI_Init(&argc, &argv);
  if (err != MPI_SUCCESS) {
    cout << "\nError initializing MPI\n";
    MPI_Abort(MPI_COMM_WORLD, err);
  }

  MPI_Comm_size(MPI_COMM_WORLD, &procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  // ------ PARAMETER SETUP -----------
  pow_2 = atoi(argv[1]);
  max_iter = atoi(argv[2]);

  n = pow(2, pow_2);
  arr = new char[n];

  if (myid == 0) {
    for (int j = 0; j < n; j++)
      arr[j] = 0;
  }
    // Warmup
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(arr, n, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    //main measurement
    if (myid == 0)
      t_start = MPI_Wtime();

    for (auto iter = 0; iter < max_iter; iter++) {
      MPI_Bcast(arr, n, MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (myid == 0) {
      t_end = MPI_Wtime();
      mpi_time = (t_end - t_start) * SCALE;
    }
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  if (myid == 0) {
    // cout << "\nData Size: " << nBytes << " bytes\n";
    cout << mpi_time << "\n";
    // cout << "Mean of communication times: " << Mean(mpi_time , num_restart)
    //      << "\n";
    // cout << "Median of communication times: " << Median(mpi_time , num_restart )
    //      << "\n";
    // Print_times(mpi_time, num_restart);
  }
  free(arr);
  MPI_Finalize();
  return 0;
} // end main

