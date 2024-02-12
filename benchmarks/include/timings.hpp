#pragma once
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <tuple>
#include <vector>

struct benchmark_timer {
    constexpr benchmark_timer()
        : mpi_time{0, std::numeric_limits<double>::min()}, compact_time{0, std::numeric_limits<double>::min()},
          view_time{0, std::numeric_limits<double>::min()} {}

    auto get_timings() const {
        return std::make_tuple(
            mpi_time[1] - mpi_time[0], compact_time[1] - compact_time[0], view_time[1] - view_time[0]);
    }

    auto operator()() const { return get_timings(); }

    static constexpr short start = 0;
    static constexpr short end = 1;

    double mpi_time[2];
    double compact_time[2];
    double view_time[2];
};

struct time_consumer {
    time_consumer() = default;

    void add(benchmark_timer t) { m_times.push_back(t); }

    void emit_results(std::vector<double> &times, std::string metric_name) const {
        // Calculate median
        const size_t size = times.size();
        std::sort(times.begin(), times.end());
        
        const double median_time = size % 2 == 0 ? (times[size / 2 - 1] + times[size / 2]) / 2 : times[size / 2];
        const double max_time = *(times.end() - 1);
        const double min_time = *times.begin();

        // Calculate standard deviation
        const double mean = std::accumulate(times.begin(), times.end(), 0.0) / size;
        const double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
        const double stdev_time = std::sqrt(sq_sum / size - mean * mean);

        std::cout << metric_name << "\n";
        std::cout << "\t- Maximum: " << max_time <<"s\n"; 
        std::cout << "\t- Minimum: " << min_time <<"s\n"; 
        std::cout << "\t- Median: " << median_time <<"s\n"; 
        std::cout << "\t- Standard deviation: " << stdev_time <<"s\n"; 
    }

    void consume_results() const {
        std::vector<double> mpi_times;
        std::vector<double> compact_times;
        std::vector<double> view_times;

        // Todo this should be generalized
        std::for_each(m_times.cbegin(), m_times.cend(), [&](const benchmark_timer &t) {
            auto [mpi_time, compact_time, view_time] = t.get_timings();
            mpi_times.push_back(mpi_time);
            compact_times.push_back(compact_time);
            view_times.push_back(view_time);
        });
        
        const auto default_precision{std::cout.precision()};
        std::cout << std::setprecision(3);
        emit_results(mpi_times, "MPI time");
        emit_results(compact_times, "Datatype compact time");
        emit_results(view_times, "Datatype build time");
        std::cout << std::setprecision(default_precision);
        std::cout << "---------------------------" << std::endl;

    }


  private:
    std::vector<benchmark_timer> m_times;
    static constexpr size_t scale = 1000000;
};