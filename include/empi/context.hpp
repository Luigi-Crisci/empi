/*
 * Copyright (c) 2022-2023 University of Salerno, Italy. All rights reserved.
 */

#ifndef EMPI_PROJECT_CONTEXT_HPP
#define EMPI_PROJECT_CONTEXT_HPP

#include <cstddef>
#include <memory>
#include <mpi.h>

#include "message_group.hpp"
#include <empi/message_grp_hdl.hpp>
#include <empi/tag.hpp>
#include <empi/type_traits.hpp>
#include <functional>


namespace empi {

class Context {
  public:
    Context(int *argc, char ***argv) { MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &thread_support); }

    Context(const Context &c) = delete;
    Context(Context &&c) = default;

    ~Context() {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
    }

    std::unique_ptr<MessageGroup> create_message_group(
        MPI_Comm comm, size_t pool_size = request_pool::default_pool_size) {
        return std::make_unique<MessageGroup>(comm);
    }

  private:
    int _rank;
    int thread_support;
};

#ifdef TEST
static Context &get_context() {
    static Context ctx(nullptr, nullptr);
    return ctx;
}
#endif

}; // namespace empi

#endif // __EMPI_CONTEXT_H__
