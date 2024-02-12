#pragma once

#include <mpi.h>

namespace empi{

    static double wtime() {
        return MPI_Wtime();
    }
    
    // class time {
    // public:
    //     time() = default;
    //     ~time() = default;

    //     static double wtime() {
    //         return MPI_Wtime();
    //     }
    // };

    // class timer {
    // public:
    //     timer() = default;
    //     ~timer() = default;

    //     void start() {
    //         m_start_time = time::wtime();
    //     }

    //     void stop() {
    //         m_end_time = time::wtime();
    //     }

    //     double elapsed() const {
    //         return m_end_time - m_start_time;
    //     }

    // private:
    //     double m_start_time;
    //     double m_end_time;
    // };

    
}