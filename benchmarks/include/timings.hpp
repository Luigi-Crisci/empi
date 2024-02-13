#pragma once
#include "utils.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mpi.h> // Include MPI Reduce
#include <numeric>
#include <tuple>
#include <vector>

enum time_scale { seconds = 1, milliseconds = 1000, microseconds = 1000000, nanoseconds = 1000000000 };

static auto string_to_time_scale(std::string scale) {
    std::transform(scale.begin(), scale.end(), scale.begin(), ::tolower);
    if(scale == "s" || scale == "seconds") { return time_scale::seconds; }
    if(scale == "ms" || scale == "milliseconds") { return time_scale::milliseconds; }
    if(scale == "us" || scale == "microseconds") { return time_scale::microseconds; }
    if(scale == "ns" || scale == "nanoseconds") { return time_scale::nanoseconds; }
    throw std::runtime_error("Invalid time scale");
}
template<bool Acronym = false>
static auto time_scale_to_string(time_scale scale) {
    switch(scale) {
        case time_scale::seconds: return Acronym ? "s" : "seconds";
        case time_scale::milliseconds: return Acronym ? "ms" : "milliseconds";
        case time_scale::microseconds: return Acronym ? "us" : "microseconds";
        case time_scale::nanoseconds: return Acronym ? "ns" : "nanoseconds";
    }
    throw std::runtime_error("Invalid time scale");
}


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
    time_consumer(time_scale scale = time_scale::seconds) : m_scale(scale) {}

    void add(benchmark_timer t) { m_times.push_back(t); }

    void set_time_scale(time_scale scale) { m_scale = scale; }

    void emit_results(std::vector<double> &old_times, std::string metric_name) const {
        // Obtain the slowest processor for each iteration
        std::vector<double> times(old_times.size());
        for(int i = 0; i < old_times.size(); i++) {
            MPI_Reduce(&old_times[i], &times[i], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        }

        if(is_master()) {
            // Calculate median
            const size_t size = times.size();
            std::sort(times.begin(), times.end());
            double min_time = *times.begin();
            double max_time = *(times.end() - 1);
            double median_time = size % 2 == 0 ? (times[size / 2 - 1] + times[size / 2]) / 2 : times[size / 2];

            // Calculate standard deviation
            double mean = std::accumulate(times.begin(), times.end(), 0.0) / size;
            double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
            double stdev_time = std::sqrt(sq_sum / size - mean * mean);

            std::cout << metric_name << ":\n";
            const auto print_time_scaled = [&](double time, std::string name) {
                std::cout << "\t- " << name << ": " << time * static_cast<double>(m_scale)
                          << time_scale_to_string<true>(m_scale) << "\n";
            };
            print_time_scaled(max_time, "Maximum");
            print_time_scaled(min_time, "Minimum");
            print_time_scaled(mean, "Mean");
            print_time_scaled(median_time, "Median");
            print_time_scaled(stdev_time, "Standard deviation");
        }
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


        std::cout << std::fixed << std::setprecision(8);
        emit_results(mpi_times, "MPI time");
        emit_results(compact_times, "Datatype compact time");
        emit_results(view_times, "Datatype build time");
        if(is_master()) { std::cout << "---------------------------" << std::endl; }
    }


  private:
    std::vector<benchmark_timer> m_times;
    time_scale m_scale;
};