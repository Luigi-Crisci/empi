#include "empi/compact.hpp"
#include <catch2/catch_test_macros.hpp>

#include <cstdint>

#include <empi/empi.hpp>

namespace stdex = Kokkos;

template<typename T, class Extents, class Layout>
    requires(Extents::rank() == 1)
void print(const Kokkos::mdspan<T, Extents, Layout> &view) {
    std::cout << "\t\t";
    for(int i = 0; i < view.extent(0); ++i) { std::cout << view[i] << ", "; }
}

template<typename T, class Extents>
    requires(Extents::rank() == 2)
void print(const Kokkos::mdspan<T, Extents> &view) {
    for(int i = 0; i < view.extent(0); ++i) {
        if(i != 0) std::cout << "\n";
        std::cout << "\t";
        for(int j = 0; j < view.extent(1); ++j) { std::cout << view(i, j) << ", "; }
    }
}

TEST_CASE("Create column mdspan", "[column_layout]") {
    std::vector<int> v(10);
    std::iota(v.begin(), v.end(), 0);

    Kokkos::extents<size_t, 5, 2> ext;
    Kokkos::mdspan<int, decltype(ext)> tmp(v.data(), ext);
    auto view = empi::layouts::column_layout::build(v, ext, 1);

    REQUIRE(decltype(view)::rank() == 1);
    REQUIRE(view.extent(0) == 5);

    for(int i = 1, j = 0; i < 10; i += 2, j++) REQUIRE(v[i] == view[j]);
}


TEST_CASE("Check extents type traits", "[mdspan]") {
    Kokkos::extents<int, 1, 2, 3, 4, 5> ex;
    using trimmed_extent = decltype(empi::details::remove_last(ex));
    REQUIRE(std::is_same_v<trimmed_extent, Kokkos::extents<int, 1, 2, 3, 4>>);
}

struct S {
    int x;
    int y;
};

TEST_CASE("Create struct layout", "[mdspan|struct_layout]") {
    std::vector<S> tmp(10);
    int count = 0;
    std::transform(tmp.begin(), tmp.end(), tmp.begin(), [&count](S &s) {
        s.x = count++;
        s.y = count++;
        return s;
    });
    auto proj = [](S &s) -> int & { return s.y; };
    auto ext = Kokkos::extents(10);
    std::array stride{2};
    Kokkos::mdspan view{tmp.data(), Kokkos::layout_stride::mapping{ext, stride},
        empi::layouts::struct_layout::struct_accessor<S, decltype(proj)>(std::move(proj))};

    for(int i = 0; i < 5; i += 1) {
        // REQUIRE(view[i] == i*2+1);
        // std::cout << view[i] << "\n";
    }
}

TEST_CASE("Compact column layout", "[mdspan|layouts|compact]") {
    std::vector<int> v(10);
    std::iota(v.begin(), v.end(), 0);
    Kokkos::extents<size_t, 5, 2> ext;
    Kokkos::mdspan<int, decltype(ext)> tmp(v.data(), ext);
    auto view = empi::layouts::column_layout::build(v, ext, 1);

    auto ptr = empi::layouts::compact(view);

    for(int i = 0; i < view.size(); i++) { REQUIRE(ptr.get()[i] == view(i)); }
}

TEST_CASE("Create tiled block layout", "[mdspan|layouts|compact]") {
    std::vector<int> v(10);
    std::iota(v.begin(), v.end(), 0);

    Kokkos::dextents<size_t, 1> ext(10 / 5 * 3);
    size_t blocks{3};
    size_t strides{5};
    auto view = empi::layouts::block_layout::build(v, ext, blocks, strides, Kokkos::default_accessor<int>());
    for(int i = 0; i < 6; i++) { REQUIRE(view[i] == i + (2 * (i / 3))); }
}

TEST_CASE("Create blocked block layout", "[mdspan|layouts|compact]") {
    std::vector<int> v(20);
    std::iota(v.begin(), v.end(), 0);

    Kokkos::extents<size_t, 20> ext;
    size_t blocks{2};
    std::array<size_t, 2> strides({3, 5});
    auto view = empi::layouts::block_layout::build(v, ext, blocks, strides);
    REQUIRE(view[0] == 0);
    REQUIRE(view[1] == 1);
    REQUIRE(view[2] == 3);
    REQUIRE(view[3] == 4);
    REQUIRE(view[4] == 8);
    REQUIRE(view[5] == 9);
    REQUIRE(view[6] == 11);
    REQUIRE(view[7] == 12);
    REQUIRE(view[8] == 16);
    REQUIRE(view[9] == 17);
}

TEST_CASE("Create bucket block layout", "[mdspan|layouts|compact]") {
    std::vector<int> v(20);
    std::iota(v.begin(), v.end(), 0);

    Kokkos::extents<size_t, 20> ext;
    std::array<size_t, 2> blocks({4, 3});
    size_t strides{5};
    auto view = empi::layouts::block_layout::build(v, ext, blocks, strides, Kokkos::default_accessor<int>());
    REQUIRE(view[0] == 0);
    REQUIRE(view[1] == 1);
    REQUIRE(view[2] == 2);
    REQUIRE(view[3] == 3);
    REQUIRE(view[4] == 5);
    REQUIRE(view[5] == 6);
    REQUIRE(view[6] == 7);
    REQUIRE(view[7] == 10);
    REQUIRE(view[8] == 11);
    REQUIRE(view[9] == 12);
    REQUIRE(view[10] == 13);
    REQUIRE(view[11] == 15);
    REQUIRE(view[12] == 16);
    REQUIRE(view[13] == 17);
}

TEST_CASE("Create alternating block layout", "[mdspan|layouts|compact]") {
    std::vector<int> v(20);
    std::iota(v.begin(), v.end(), 0);

    Kokkos::extents<size_t, 20> ext;
    std::array<size_t, 2> blocks({4, 3});
    std::array<size_t, 2> strides({5, 6});
    auto view = empi::layouts::block_layout::build(v, ext, blocks, strides, Kokkos::default_accessor<int>());

    REQUIRE(view[0] == 0);
    REQUIRE(view[1] == 1);
    REQUIRE(view[2] == 2);
    REQUIRE(view[3] == 3);
    REQUIRE(view[4] == 5);
    REQUIRE(view[5] == 6);
    REQUIRE(view[6] == 7);
    REQUIRE(view[7] == 11);
    REQUIRE(view[8] == 12);
    REQUIRE(view[9] == 13);
    REQUIRE(view[10] == 14);
    REQUIRE(view[11] == 16);
    REQUIRE(view[12] == 17);
    REQUIRE(view[13] == 18);
}
