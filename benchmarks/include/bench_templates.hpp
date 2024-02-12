#pragma once

#include <empi/empi.hpp>
#include <iostream>



template<typename T>
struct mpi_benchmark{
    using value_type = T;

    mpi_benchmark(int argc, char **argv) {
        int err = MPI_Init(&argc, &argv);
        if(err != MPI_SUCCESS) {
            std::cerr << "\nError initializing MPI.\n";
            MPI_Abort(MPI_COMM_WORLD, err);
        }
    }

    ~mpi_benchmark() { MPI_Finalize(); }

    static std::string get_name(){
        return EMPI_BENCHMARK_NAME;
    }
};
    
template<typename T>
struct empi_benchmark{
    using value_type = T;

    empi_benchmark(int argc, char **argv) : m_ctx(&argc, &argv) {
        m_message_group = m_ctx.create_message_group(MPI_COMM_WORLD);
    }

    ~empi_benchmark() = default;

    static std::string get_name(){
        return EMPI_BENCHMARK_NAME;
    }

    empi::Context m_ctx;
    std::unique_ptr<empi::MessageGroup> m_message_group;
};

