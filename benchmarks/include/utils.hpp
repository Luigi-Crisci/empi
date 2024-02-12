#ifndef EXAMPLES_DATATYPES_UTILS
#define EXAMPLES_DATATYPES_UTILS

#include <algorithm>
#include <cassert>
#include <iostream>
#include <mpi.h>
#include <string>

int get_rank(){
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

bool is_master(){
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank == 0;
}

struct basic_struct {
    int x;
    int y;
    float z;

    basic_struct() : x(0), y(0), z(0.0) {}

    //define copy assignment operator
    basic_struct &operator=(const basic_struct &other) {
        x = other.x;
        y = other.y;
        z = other.z;
        return *this;
    }

    basic_struct& operator++() {
        x++;
        y++;
        z++;
        return *this;
    }
};

template<typename T>
void print_safe_mpi(T *data, int n, int rank){
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    for(int i = 0; i < size; i++) {
        if(i == rank) {
            std::cout << "Rank " << rank << ": ";
            for(int j = 0; j < n; j++) {
                std::cout << data[j] << ", ";
            }
            std::cout << "\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

template<typename T>
void print_mpi_exclusive(T* data, int n, int exec_rank, int caller_rank){
    if (caller_rank == exec_rank){
            std::cout << "Rank " << caller_rank <<": ";
            for(int j = 0; j < n; j++) {
                std::cout << data[j] << ", ";
            }
            std::cout << "\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
}


static MPI_Datatype basic_type = MPI_CHAR;

int get_communication_size(int num_bytes, MPI_Datatype derived_datatype, MPI_Datatype raw_datatype) {
    MPI_Aint derived_datatype_extent;
    MPI_Aint raw_datatype_extent;
    MPI_Aint lb;
    MPI_Type_get_extent(derived_datatype, &lb, &derived_datatype_extent);
    MPI_Type_get_extent(raw_datatype, &lb, &raw_datatype_extent);
    assert(num_bytes % raw_datatype_extent == 0);

    num_bytes = num_bytes / raw_datatype_extent;
    auto datatype_size = derived_datatype_extent / raw_datatype_extent;
    assert(num_bytes > 0);
    assert(num_bytes >= datatype_size && num_bytes % datatype_size == 0);

    std::cout << "Num elements: " << num_bytes << "\n";
    std::cout << "Datatype extent: " << derived_datatype_extent << "\n";
    std::cout << "Raw extent: " << raw_datatype_extent << "\n";

    return num_bytes / datatype_size;
}

void build_struct_mpi_type(MPI_Datatype *t) {
    int blocklengths[3] = {1, 1, 1};
    MPI_Aint displacements[3] = {0, sizeof(int), 2 * sizeof(int)};
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_FLOAT};
    MPI_Type_create_struct(3, blocklengths, displacements, types, t);
    MPI_Type_commit(t);
}

MPI_Datatype get_datatype(const std::string &type) {
    if(type == "basic") {
        return basic_type;
    } else {
        MPI_Datatype struct_t;
        build_struct_mpi_type(&struct_t);
        return struct_t;
    }
}

template<typename T>
void *allocate(int n) {
    return (void *)malloc(sizeof(T) * n);
}

double Mean(double a[], int n) {
    double sum = 0.0;
    for(int i = 0; i < n; i++) sum += a[i];

    return (sum / (double)n);
}

double Median(double a[], int n) {
    std::sort(a, a + n);
    if(n % 2 != 0) return a[n / 2];

    return (a[(n - 1) / 2] + a[n / 2]) / 2.0;
}

void Print_times(double a[], int n) {
    std::cout << "\n------------------------------------";
    for(int t = 0; t < n; t++) std::cout << "\n " << a[t];
}


/*******************************************************/
/* basic layouts */
/*******************************************************/

int bl_basetype(MPI_Datatype *t, int *flags) {
    *flags = 0;

    return MPI_SUCCESS;
}

// elements A, extent (in units of b) B
int bl_column(MPI_Datatype *t, int col_size, int row_size,  MPI_Datatype b) {
    MPI_Type_vector(col_size, 1, row_size, b, t);
    MPI_Type_commit(t);
    return MPI_SUCCESS;
}



// elements A, extent (in units of b) B
int bl_tiled(MPI_Datatype *t, int *flags, int A, int B, MPI_Datatype b) {
    MPI_Datatype t1;
    MPI_Aint lb, eb;

    *flags = 0;

    assert(A > 0);
    assert(A <= B); // we allow degenerate case with A==B

    MPI_Type_get_extent(b, &lb, &eb); // get extent of basetype
    MPI_Type_contiguous(A, b, &t1);
    MPI_Type_create_resized(t1, 0, B * eb, t);
    MPI_Type_commit(t);
    MPI_Type_free(&t1); // make sure intermediate type is eventually freed

    return MPI_SUCCESS;
}


// elements A1+A2, extent (in units of b) 2*B
int bl_bucket(MPI_Datatype *t, int *flags, MPI_Datatype b, int A1, int A2, int B) {
    MPI_Datatype t1;
    MPI_Aint lb, eb;
    int block[2], displ[2];

    *flags = 0;

    assert(A1 > 0);
    assert(A2 > 0); // we allow degerate case with A1==A2
    assert(B >= A1 && B >= A2);

    block[0] = A1;
    block[1] = A2;
    displ[0] = 0;
    displ[1] = B;

    MPI_Type_get_extent(b, &lb, &eb); // get extent of basetype
    MPI_Type_indexed(2, block, displ, b, &t1);
    MPI_Type_create_resized(t1, 0, 2 * B * eb, t);
    MPI_Type_free(&t1); // make sure intermediate type is eventually freed
    MPI_Type_commit(t);

    return MPI_SUCCESS;
}


// elements 2*A, extent (in units of b) B1+B2
int bl_block(MPI_Datatype *t, int *flags, MPI_Datatype b, int A, int B1, int B2) {
    MPI_Datatype t1;
    MPI_Aint lb, eb;

    int displ[2];

    *flags = 0;

    assert(A > 0);
    assert(B1 >= A && B2 >= A);

    displ[0] = 0;
    displ[1] = B1;

    MPI_Type_get_extent(b, &lb, &eb); // get extent of basetype
    MPI_Type_create_indexed_block(2, A, displ, b, &t1);
    MPI_Type_create_resized(t1, 0, (B1 + B2) * eb, t);
    MPI_Type_free(&t1); // make sure intermediate type is eventually freed
    MPI_Type_commit(t);

    return MPI_SUCCESS;
}


// elements A1+A2, corresponding extents (in units of b) B1 and B1
int bl_alternating(MPI_Datatype *t, int *flags, MPI_Datatype b, int A1, int A2, int B1, int B2) {
    MPI_Datatype t1;
    MPI_Aint lb, eb;
    int block[2], displ[2];

    *flags = 0;

    assert(A1 > 0);
    assert(A2 > 0); // we allow degenerate case with A1==A2
    assert(B1 >= A1 && B2 >= A2);

    block[0] = A1;
    block[1] = A2;
    displ[0] = 0;
    displ[1] = B1;

    MPI_Type_get_extent(b, &lb, &eb); // get extent of basetype
    MPI_Type_indexed(2, block, displ, b, &t1);
    MPI_Type_create_resized(t1, 0, (B1 + B2) * eb, t);
    MPI_Type_free(&t1); // make sure intermediate type is eventually freed
    MPI_Type_commit(t);

    return MPI_SUCCESS;
}



#endif /* EXAMPLES_DATATYPES_UTILS */
