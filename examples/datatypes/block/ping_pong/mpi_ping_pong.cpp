#include <bits/stdc++.h>
#include <chrono>
#include <iostream>
#include <malloc.h>
#include <mpi.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#include "../../utils.hpp"

using namespace std;

double Mean(double[], int);
double Median(double[], int);
void Print_times(double[], int);

int main(int argc, char **argv) {
  int myid, procs, n, err, max_iter, nBytes, sleep_time, iter = 0, range = 100,
                                                         pow_2;
  double t_start, t_end;
  constexpr int SCALE = 1000000;

  MPI_Status status;

  err = MPI_Init(&argc, &argv);
  if (err != MPI_SUCCESS) {
    cout << "\nError initializing MPI.\n";
    MPI_Abort(MPI_COMM_WORLD, err);
  }

  MPI_Comm_size(MPI_COMM_WORLD, &procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  // ------ PARAMETER SETUP -----------
  pow_2 = atoi(argv[1]);
  max_iter = atoi(argv[2]);

  double mpi_time;
  nBytes = pow(2, pow_2);
  n = nBytes;

  // Getting layout values
  int A = std::stoi(argv[3]);
  int B1 = std::stoi(argv[4]);
  int B2 = std::stoi(argv[5]);
  string datatype = argv[6];
  MPI_Datatype tiled_datatype;
  int flags;

  void *myarr, *arr;
  if (datatype == "basic") {
    myarr = allocate<char>(n / sizeof(char));
    arr = allocate<char>(n / sizeof(char));
  } else {
    myarr = allocate<basic_struct>(n / sizeof(basic_struct));
    arr = allocate<basic_struct>(n / sizeof(basic_struct));
  }

  auto raw_datatype = get_datatype(datatype);
  bl_block(&tiled_datatype, &flags, raw_datatype, A, B1, B2);

  MPI_Aint lb, size, extent;
  MPI_Aint basic_size, basic_extent;
  MPI_Type_get_extent(tiled_datatype, &lb, &extent);
  MPI_Type_get_true_extent(tiled_datatype, &lb, &size);
  MPI_Type_get_extent(raw_datatype, &basic_size, &basic_extent);

  n = n / basic_extent;
  assert(n > 0);

  assert(A <= B1 && A <= B2);
  assert(n % (B1 + B2) == 0);
  auto num_blocks = (n / (B1 + B2)) * 2;
  auto tiled_size = num_blocks * A / (B1 + B2);
  // if (myid == 0) {
  //   std::cout << "tiled size: " << tiled_size << "\n";
  //   std::cout << "Extent: " << extent << "\n";
  //   std::cout << "Size: " << size << "\n";
  //   std::cout << "Total size: " << tiled_size * size << "\n";
  //   std::cout << "Required memory: " << tiled_size * extent << "\n";
  // }

  // Warmup
  MPI_Barrier(MPI_COMM_WORLD);
  if (myid == 0) {
    MPI_Send(arr, tiled_size, tiled_datatype, 1, 0, MPI_COMM_WORLD);
    MPI_Recv(arr, tiled_size, tiled_datatype, 1, MPI_ANY_TAG, MPI_COMM_WORLD,
             &status);
  } else { // Node rank 1
    MPI_Recv(myarr, tiled_size, tiled_datatype, 0, MPI_ANY_TAG, MPI_COMM_WORLD,
             &status);
    MPI_Send(myarr, tiled_size, tiled_datatype, 0, 1, MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  if (myid == 0)
    t_start = MPI_Wtime();

  for (auto iter = 0; iter < max_iter; iter++) {
    if (myid == 0) {
      MPI_Send(arr, tiled_size, tiled_datatype, 1, 0, MPI_COMM_WORLD);
      MPI_Recv(arr, tiled_size, tiled_datatype, 1, MPI_ANY_TAG, MPI_COMM_WORLD,
               &status);
    } else { // Node rank 1
      MPI_Recv(myarr, tiled_size, tiled_datatype, 0, MPI_ANY_TAG,
               MPI_COMM_WORLD, &status);
      MPI_Send(myarr, tiled_size, tiled_datatype, 0, 1, MPI_COMM_WORLD);
    }
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
    // cout << "Median of communication times: " << Median(mpi_time ,
    // num_restart )
    //      << "\n";
    // Print_times(mpi_time, num_restart);
  }
  free(arr);
  free(myarr);
  MPI_Finalize();
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