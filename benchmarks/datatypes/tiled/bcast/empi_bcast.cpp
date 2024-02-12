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

constexpr int WARMUP = 100;


template<typename T>
void run_bench(std::shared_ptr<empi::MessageGroup> message_group, benchmark_args &args) {
    using type = T;
    size_t A = std::stoi(args.argv[3]);
    size_t B = std::stoi(args.argv[4]);
    size_t n = args.n;
    size_t max_iter = args.max_iter;
    timings &t = args.times;

    assert(B >= A);
    n = n / sizeof(type);

    assert(n > 0);

    auto view_size = n / B * A;
    std::vector<type> myarr(n);
    if(message_group->rank() == 0) { std::iota(myarr.begin(), myarr.end(), T{}); }

    t.view_time[timings::start] = empi::wtime();
    Kokkos::dextents<size_t, 1> ext(view_size);
    auto view = empi::layouts::block_layout::build(myarr, ext, A, B);
    t.view_time[timings::end] = empi::wtime();

    message_group->run([&](empi::MessageGroupHandler<type, empi::Tag{0}, empi::NOSIZE> &mgh) {
        // Warmup
        // mgh.barrier();
        // for(auto iter = 0; iter < WARMUP; iter++) { mgh.Bcast(view, 0, view_size); }
        // mgh.barrier();

        t.mpi_time[timings::start] = empi::wtime();
        // auto &&ptr = empi::layouts::block_layout::compact(view);
        auto &&ptr = empi::layouts::compact(view); // basic compact function
        t.compact_time = MPI_Wtime() - t.mpi_time[timings::start];

        for(auto iter = 0; iter < max_iter; iter++) { mgh.Bcast(ptr.get(), 0, view_size); }

        message_group->barrier();
        t.mpi_time[timings::end] = empi::wtime();
        print_safe_mpi(ptr.get(), view_size, message_group->rank()); 
        // Verify
        // for(auto i = 0; i < view.size(); i++) {
        //     if(ptr.get()[i] != i) {
        //         std::cerr << "Error at index " << i << " value: " << view[i] << std::endl;
        //         std::exit(-1);
        //     }
        // }
    });

    

}

int main(int argc, char **argv) {
    empi::Context ctx(&argc, &argv);
    std::string datatype = argv[5];

    // ------ PARAMETER SETUP -----------
    benchmark_args args{argc, argv};

    auto message_group_tmp = ctx.create_message_group(MPI_COMM_WORLD);
    auto message_group = std::make_shared<empi::MessageGroup>(std::move(*message_group_tmp));


    if(datatype == "basic") {
        run_bench<unsigned char>(message_group, args);
    // } else if(datatype == "struct") {
    //     run_bench<basic_struct>(message_group, args);
    } else {
        std::cerr << "Invalid datatype: " << datatype << std::endl;
        return 1;
    }

    message_group->barrier();
    if(message_group->rank() == 0) { args.times.print(); }
    message_group_tmp.release();
    return 0;
} // end main
