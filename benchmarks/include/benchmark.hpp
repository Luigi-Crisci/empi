#pragma once

#include "mpi.h"
#include "timings.hpp"
#include "utils.hpp"
#include <argparse/argparse.hpp>

struct benchmark_args {
    benchmark_args(size_t size, size_t iterations, argparse::ArgumentParser &parser, int argc, char **argv)
        : size(size), iterations(iterations), parser(parser) {}
    
    benchmark_args(const benchmark_args &args) = delete;

    size_t size;
    size_t iterations;
    benchmark_timer times{};
    argparse::ArgumentParser &parser;
};

template<typename Benchmark>
class benchmark_manager {
  public:
    benchmark_manager(int argc, char **argv, std::string name) : m_parser(name), m_argc(argc), m_argv(argv) {
        m_parser.add_argument("-s", "--size").help("Number of elements to send").scan<'i', size_t>().required();

        m_parser.add_argument("-i", "--iterations")
            .help("Number of kernel iterations to run. NOTE: kernel iterations are aggregated")
            .scan<'i', size_t>()
            .default_value<size_t>(100);

        m_parser.add_argument("-wr", "--warmup-runs")
            .help("Number of warmup runs")
            .default_value<size_t>(10)
            .scan<'i', size_t>();

        m_parser.add_argument("--runs").help("Number of runs to perform").scan<'i', size_t>().default_value<size_t>(5);
        m_parser.add_argument("--scale")
            .help("Scale to use for time measurements. Default is seconds")
            .action([](const std::string &value) { return string_to_time_scale(value); })
            .default_value(time_scale::seconds);
    }

    argparse::ArgumentParser &get_parser() { return m_parser; }


    void run_benchmark() {
        try {
            m_parser.parse_args(m_argc, m_argv);
        } catch(const std::exception &err) {
            std::cerr << err.what() << std::endl;
            std::cerr << m_parser;
            std::exit(1);
        }
        const auto size = m_parser.get<size_t>("-s");
        const auto iterations = m_parser.get<size_t>("-i");
        const auto runs = m_parser.get<size_t>("--runs");
        const auto warmup_runs = m_parser.get<size_t>("--warmup-runs");
        const auto scale = m_parser.get<time_scale>("--scale");
        m_times.set_time_scale(scale);

        Benchmark m_benchmark{m_argc, m_argv};
        if(is_master()) {
            std::cout << "###### Benchmark: " << Benchmark::get_name() << " ######" << std::endl;
            std::cout << "Configuration: " << std::endl;
            std::cout << "\t- Num elements: " << size << std::endl;
            std::cout << "\t- Iterations: " << iterations << std::endl;
            std::cout << "\t- Runs: " << runs << std::endl;
            std::cout << "\t- Warmup runs: " << warmup_runs << std::endl;
            std::cout << "\t- Time scale: " << time_scale_to_string(scale) << std::endl;
            std::cout << "---------------------------" << std::endl;
        }
        for(auto i = 0; i < runs + warmup_runs; i++) {
            benchmark_args args = {size, iterations, m_parser, m_argc, m_argv};
            MPI_Barrier(MPI_COMM_WORLD); // Ensure all ranks start at the same time (hopefully)
            m_benchmark.run(args);
            if(i >= warmup_runs) { // Skip warmup runs
                m_times.add(args.times);
            }
        }

        // MPI_Finalize does not specify how many threads will run after it is called
        // therefore we need to ensure that just the main thread calls the consume_results
        m_times.consume_results();
    }


  private:
    int m_argc;
    char **m_argv;
    argparse::ArgumentParser m_parser;
    time_consumer m_times;
};
