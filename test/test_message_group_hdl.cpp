#include "utils.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <empi/empi.hpp>
#include <empi/utils.hpp>
#include <experimental/mdspan>

namespace stdex = Kokkos;

TEST_CASE("Send and receive a view with no compile-time parameters", "[mgh][layouts]") {
    auto &ctx = empi::get_context();
    auto mg = ctx.create_message_group(MPI_COMM_WORLD);

    std::vector<int> v(10);
    if(mg->rank() == 0) std::iota(v.begin(), v.end(), 0);

    mg->run([&](empi::MessageGroupHandler<float> &mgh) {
        if(mg->rank() == 0) {
            Kokkos::mdspan<int, Kokkos::dextents<int, 1>> view(v.data(), 10);
            mgh.send(view, 1, 10, empi::Tag{1});
        } else {
            MPI_Status s;
            mgh.recv(v, 0, 10, empi::Tag{1}, s);

            for(int i = 0; i < 10; i++) { REQUIRE(v[i] == i); }
        }
    });
}


TEST_CASE("Send and receive a column of a matrix", "[mgh][layouts]") {
    auto &ctx = empi::get_context();
    auto mg = ctx.create_message_group(MPI_COMM_WORLD);

    std::vector<int> v(16);
    if(mg->rank() == 0) std::iota(v.begin(), v.end(), 0);

    mg->run([&](empi::MessageGroupHandler<float> &mgh) {
        if(mg->rank() == 0) {
            auto view = empi::layouts::column_layout::build(v, Kokkos::dextents<int, 2>(4, 4), 3);
            mgh.send(view, 1, 4, empi::Tag{1});
        } else {
            MPI_Status s;
            std::vector<int> dest(4);
            mgh.recv(dest, 0, 4, empi::Tag{1}, s);

            for(int i = 0; i < 4; i++) { REQUIRE(dest[i] == i * 4 + 3); }
        }
    });
}

TEST_CASE("Send and receive a vector of struct", "[mgh][layouts]") {
    auto &ctx = empi::get_context();
    auto mg = ctx.create_message_group(MPI_COMM_WORLD);

    struct S {
        int x, y;
    };

    std::vector<S> v(16);
    if(mg->rank() == 0)
        std::transform(v.begin(), v.end(), v.begin(), [i = 0](S &s) mutable {
            s.x = i++;
            s.y = i++;
            return s;
        });

    mg->run([&](empi::MessageGroupHandler<float> &mgh) {
        if(mg->rank() == 0) {
            auto view = empi::layouts::contiguous_layout::build(v, empi::layouts::struct_layout::struct_accessor<S>());
            for(int i = 0; i < 16; i++) {
                REQUIRE(view[i].x == i * 2);
                REQUIRE(view[i].y == i * 2 + 1);
            }
            mgh.send(view, 1, 16, empi::Tag{1});
        } else {
            MPI_Status s;
            std::vector<S> dest(16);
            mgh.recv(dest, 0, 16, empi::Tag{1}, s);

            for(int i = 0; i < 16; i++) {
                REQUIRE(dest[i].x == i * 2);
                REQUIRE(dest[i].y == i * 2 + 1);
            }
        }
    });
}

TEST_CASE("Send and receive a vector of one field of a struct", "[mgh][layouts]") {
    auto &ctx = empi::get_context();
    auto mg = ctx.create_message_group(MPI_COMM_WORLD);

    struct S {
        int x, y;
    };

    std::vector<S> v(16);
    if(mg->rank() == 0)
        std::transform(v.begin(), v.end(), v.begin(), [i = 0](S &s) mutable {
            s.x = i++;
            s.y = i++;
            return s;
        });

    mg->run([&](empi::MessageGroupHandler<float> &mgh) {
        if(mg->rank() == 0) {
            auto proj = [](S &s) -> int & { return s.x; };
            auto view = empi::layouts::contiguous_layout::build(
                v, empi::layouts::struct_layout::struct_accessor<S, decltype(proj)>(std::move(proj)));
            for(int i = 0; i < 16; i++) { REQUIRE(view[i] == i * 2); }
            mgh.send(view, 1, 16, empi::Tag{1});
        } else {
            MPI_Status s;
            std::vector<int> dest(16);
            mgh.recv(dest, 0, 16, empi::Tag{1}, s);

            for(int i = 0; i < 16; i++) { REQUIRE(dest[i] == i * 2); }
        }
    });
}

TEST_CASE("Send and receive a vector of a subest of fields of a struct", "[mgh][layouts]") {
    auto &ctx = empi::get_context();
    auto mg = ctx.create_message_group(MPI_COMM_WORLD);

    struct S {
        int x, y, z;
    };

    std::vector<S> v(16);
    if(mg->rank() == 0)
        std::transform(v.begin(), v.end(), v.begin(), [i = 0](S &s) mutable {
            s.x = i++;
            s.y = i++, s.z = i++;
            return s;
        });

    mg->run([&](empi::MessageGroupHandler<float> &mgh) {
        if(mg->rank() == 0) {
            auto proj = [](S &s) -> std::tuple<int, int> { return {s.x, s.z}; };
            auto view = empi::layouts::contiguous_layout::build(
                v, empi::layouts::struct_layout::struct_accessor<S, decltype(proj)>(std::move(proj)));
            for(int i = 0; i < 16; i++) {
                REQUIRE(std::get<0>(view[i]) == i * 3);
                REQUIRE(std::get<1>(view[i]) == i * 3 + 2);
            }
            mgh.send(view, 1, 16, empi::Tag{1});
        } else {
            MPI_Status s;
            std::vector<std::tuple<int, int>> dest(16);
            mgh.recv(dest, 0, 16, empi::Tag{1}, s);

            for(int i = 0; i < 16; i++) {
                REQUIRE(std::get<0>(dest[i]) == i * 3);
                REQUIRE(std::get<1>(dest[i]) == i * 3 + 2);
            }
        }
    });
}

TEST_CASE("Send and receive a column of a subest of fields of a struct", "[mgh][layouts]") {
    auto &ctx = empi::get_context();
    auto mg = ctx.create_message_group(MPI_COMM_WORLD);

    std::vector<trivial_struct> v(36);
    if(mg->rank() == 0) {
        std::transform(v.begin(), v.end(), v.begin(), [i = 0](trivial_struct &s) mutable {
            s.x = i++;
            s.y = i++, s.z = (static_cast<float>(i) * 1.2);
            i++;
            return s;
        });
    }

    mg->run([&](empi::MessageGroupHandler<float> &mgh) {
        if(mg->rank() == 0) {
            Kokkos::extents<size_t, 6, 6> ext;
            constexpr int col = 0;
            auto acc = empi::layouts::make_struct_accessor<trivial_struct>(STRUCT_FIELDS(trivial_struct, x, z));
            auto view = empi::layouts::column_layout::build(v, ext, col, acc);
            for(int i = 0; i < 6; i++) {
                REQUIRE(std::get<0>(view[i]) == i * 3 * 6 + col * 3);
                REQUIRE(std::get<1>(view[i]) == Catch::Approx((i * 3 * 6 + col * 3 + 2) * 1.2));
            }
            mgh.send(view, 1, 6, empi::Tag{1});
        } else {
            MPI_Status s;
            std::vector<std::tuple<int, float>> dest(6);
            mgh.recv(dest, 0, 6, empi::Tag{1}, s);

            for(int i = 0; i < 6; i++) {
                REQUIRE(std::get<0>(dest[i]) == i * 3 * 6);
                REQUIRE(std::get<1>(dest[i]) == Catch::Approx((i * 3 * 6 + 2) * 1.2));
            }
        }
    });
}

TEST_CASE("Send and receive scalar values", "[mgh]") {
    auto &ctx = empi::get_context();
    auto mg = ctx.create_message_group(MPI_COMM_WORLD);

    auto tag = empi::Tag{0};
    mg->run([&](empi::MessageGroupHandler<float> &mgh) {
        // Send an int
        if(mg->rank() == 0) {
            int val = 5;
            mgh.send(val, 1, 1, tag);
        } else {
            int res;
            MPI_Status s;
            mgh.recv(res, 0, 1, tag, s);
            REQUIRE(res == 5);
        }

        // Send a pointer
        if(mg->rank() == 0) {
            int val = 5;
            mgh.send(&val, 1, 1, tag);
        } else {
            int res;
            MPI_Status s;
            mgh.recv(&res, 0, 1, tag, s);
            REQUIRE(res == 5);
        }

        // Send a string
        if(mg->rank() == 0) {
            std::string val = "hello";
            mgh.send(val, 1, val.size(), tag);
        } else {
            std::string res;
            res.resize(5);
            MPI_Status s;
            mgh.recv(res, 0, 5, tag, s);
            REQUIRE(res == "hello");
        }

        // Send a C string
        if(mg->rank() == 0) {
            char *val = "hello";
            REQUIRE(strlen(val) == strlen("hello"));
            mgh.send(val, 1, strlen("hello") + 1, tag);
        } else {
            char res[256];
            MPI_Status s;
            mgh.recv(res, 0, 6, tag, s);
            REQUIRE(strlen(res) == strlen("hello"));
            REQUIRE(strcmp(res, "hello") == 0);
        }

        // Send an int vector
        if(mg->rank() == 0) {
            std::vector<int> val(16);
            std::iota(val.begin(), val.end(), 0);
            mgh.send(val, 1, val.size(), tag);
        } else {
            int res[16];
            MPI_Status s;
            mgh.recv(res, 0, 16, tag, s);
            for(int i = 0; i < 16; i++) { REQUIRE(res[i] == i); }
        }
    });
}

TEST_CASE("Bcast scalar values", "[mgh]") {
    auto &ctx = empi::get_context();
    auto mg = ctx.create_message_group(MPI_COMM_WORLD);

    auto tag = empi::Tag{0};
    std::vector<float> v(32);
    mg->run([&](empi::MessageGroupHandler<float> &mgh) {
        if(mg->rank() == 0) std::iota(v.begin(), v.end(), 0);

        mgh.Bcast(v, 0, 32);

        for(int i = 0; i < 32; i++) { REQUIRE(v[i] == i); }
    });
}

TEST_CASE("Gatherv scalar values using range displacements", "[mgh]") {
    auto &ctx = empi::get_context();
    auto mg = ctx.create_message_group(MPI_COMM_WORLD);

    auto tag = empi::Tag{0};
    std::vector<int> v(32);
    std::iota(v.begin(), v.end(), 0 + 32 * mg->rank());

    std::vector<int> res_vector;
    res_vector.resize(32 * mg->size());

    mg->run([&](empi::MessageGroupHandler<float> &mgh) {
        const int message_size = 32;
        std::vector<int> recv_sizes;
        std::vector<int> displacements;

        for(int i = 0; i < mg->size(); i++) {
            recv_sizes.push_back(message_size);
            displacements.push_back(i * message_size);
        }

        mgh.gatherv(0, v, message_size, res_vector, recv_sizes, displacements);

        if(mg->rank() == 0) {
            for(int i = 0; i < 32 * mg->size(); i++) { REQUIRE(res_vector[i] == i); }
        }
    });
}

TEST_CASE("Gatherv scalar values using raw pointer displacements", "[mgh]") {
    auto &ctx = empi::get_context();
    auto mg = ctx.create_message_group(MPI_COMM_WORLD);

    auto tag = empi::Tag{0};
    std::vector<int> v(32);
    std::iota(v.begin(), v.end(), 0 + 32 * mg->rank());

    std::vector<int> res_vector;
    res_vector.resize(32 * mg->size());

    mg->run([&](empi::MessageGroupHandler<float> &mgh) {
        const int message_size = 32;
        std::vector<int> recv_sizes;
        std::vector<int> displacements;

        for(int i = 0; i < mg->size(); i++) {
            recv_sizes.push_back(message_size);
            displacements.push_back(i * message_size);
        }

        mgh.gatherv(0, v, message_size, res_vector, recv_sizes.data(), displacements.data());

        if(mg->rank() == 0) {
            for(int i = 0; i < 32 * mg->size(); i++) { REQUIRE(res_vector[i] == i); }
        }
    });
}


TEST_CASE("Allgather scalar values", "[mgh]") {
    auto &ctx = empi::get_context();
    auto mg = ctx.create_message_group(MPI_COMM_WORLD);

    auto tag = empi::Tag{0};
    std::vector<int> v(32);
    std::fill(v.begin(), v.end(), mg->rank());

    std::vector<int> res_vector;
    res_vector.resize(32);

    mg->run([&](empi::MessageGroupHandler<int> &mgh) {
        const int message_size = 32 / mg->size();
        mgh.Allgather(v, message_size, res_vector, message_size);

        for(int i = 0; i < mg->size(); i++)
            for(int j = i * message_size; j < (i * message_size) + message_size; j++) { REQUIRE(res_vector[j] == i); }
    });
}

TEST_CASE("Allgather column view", "[mgh]") {
    auto &ctx = empi::get_context();
    auto mg = ctx.create_message_group(MPI_COMM_WORLD);

    auto tag = empi::Tag{0};
    std::vector<int> v(36);
    std::fill(v.begin(), v.end(), mg->rank());
    const int message_size = 6;

    std::vector<int> res_vector;
    res_vector.resize(message_size * mg->size());

    auto ext = Kokkos::dextents<size_t, 2>(6, 6);
    auto view = empi::layouts::column_layout::build(v, ext, 3);

    // std::cout << view.size() << '\n';
    auto ptr = empi::layouts::compact(view);

    mg->barrier();
    mg->run([&](empi::MessageGroupHandler<int> &mgh) {
        mgh.Allgather(view, message_size, res_vector, message_size);

        for(int i = 0; i < mg->size(); i++)
            for(int j = i * message_size; j < (i * message_size) + message_size; j++) {
                if(mg->rank() == 0) REQUIRE(res_vector[j] == i);
            }
    });
}

TEST_CASE("Allgather block view", "[mgh]") {
    auto &ctx = empi::get_context();
    auto mg = ctx.create_message_group(MPI_COMM_WORLD);

    auto tag = empi::Tag{0};
    std::vector<char> v(32, mg->rank() + 'a');
    size_t A = 4;
    size_t B = 4;
    size_t message_size = 32 / B * A;

    std::vector<char> res_vector(message_size * mg->size());

    // if (mg->rank() == 0){
    // 	std::cout << "n: " << 32 << '\n';
    // 	std::cout << "A: " << A << '\n';
    // 	std::cout << "B: " << B << '\n';
    // 	std::cout << "View size: " << message_size << '\n';
    // 	std::cout << "Res size: " << res_vector.size() << '\n';
    // }

    auto view = empi::layouts::block_layout::build(v, Kokkos::dextents<size_t, 1>(message_size), A, B);

    // auto ptr = empi::layouts::block_layout::compact(view);
    // mg->barrier();
    mg->run([&](empi::MessageGroupHandler<char> &mgh) {
        mgh.Allgather(view, message_size, res_vector, message_size);

        for(int i = 0; i < mg->size(); i++)
            for(int j = i * message_size; j < (i * message_size) + message_size; j++) {
                // if(mg->rank() == 1)
                // std::cout << i << ": " << res_vector[j] << "\n";
                REQUIRE(res_vector[j] == i + 'a');
            }
    });
}