/*
 * Copyright (c) 2022-2023 University of Salerno, Italy. All rights reserved.
 */

#ifndef INCLUDE_EMPI_DEFINES
#define INCLUDE_EMPI_DEFINES


#if defined(ENABLE_UNCHECKED_FUNCTION)
#define EMPI_SEND MPI_USend
#define EMPI_ISEND MPI_IUsend
#define EMPI_RECV MPI_URecv
#define EMPI_IRECV MPI_IUrecv
#define EMPI_ALLREDUCE MPI_UAllreduce
#define EMPI_BCAST MPI_UBcast
#define EMPI_IBCAST MPI_IUbcast
#define EMPI_GATHERV MPI_Gatherv     // Not yet implemented
#define EMPI_ALLGATHER MPI_Allgather // Not yet implemented
#define EMPI_CHECKCOMM(comm) MPI_Checkcomm(comm)
#define EMPI_CHECKTYPE(type) MPI_Checktype(type)
#else
#define EMPI_SEND MPI_Send
#define EMPI_ISEND MPI_Isend
#define EMPI_RECV MPI_Recv
#define EMPI_IRECV MPI_Irecv
#define EMPI_ALLREDUCE MPI_Allreduce
#define EMPI_BCAST MPI_Bcast
#define EMPI_IBCAST MPI_Ibcast
#define EMPI_GATHERV MPI_Gatherv
#define EMPI_ALLGATHER MPI_Allgather
#define EMPI_CHECKCOMM(comm) // Disable function
#define EMPI_CHECKTYPE(type) // Disable function
#endif

namespace empi {


constexpr int NOSIZE = 0;

namespace details {
enum mpi_function { send = 1, isend, recv, irecv, bcast, ibcast, allreduce, gatherv, all };

template<mpi_function f>
concept is_send = requires { f == mpi_function::send || f == mpi_function::isend; };

template<mpi_function f>
concept is_recv = requires { f == mpi_function::recv || f == mpi_function::irecv; };

template<mpi_function f>
concept is_all = requires { f == mpi_function::all; };
} // namespace details
} // namespace empi

#endif /* INCLUDE_EMPI_DEFINES */
