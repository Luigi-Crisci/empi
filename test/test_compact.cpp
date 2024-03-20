#include <catch2/catch_test_macros.hpp>
#include <empi/compact/compact.hpp>
#include <empi/empi.hpp>
#include <random>

#include "utils.hpp"

namespace stdex = Kokkos;

TEST_CASE("Call compact on a trivial layout and accessor does not produce copies", "[compact][layouts]") {
    using namespace Kokkos;
    std::vector<int> v(16);
    auto view = empi::layouts::contiguous_layout::build(v);
    REQUIRE(empi::layouts::is_trivial_view<decltype(view)>);
    auto ptr = empi::layouts::compact(view);

    REQUIRE(empi::details::is_same_template_v<empi::details::conditional_deleter<int>, decltype(ptr)::deleter_type>);
}

TEST_CASE("Call compact on a contiguous layout and non-trivial accessor produces a copy", "[compact][layouts]") {
    using namespace Kokkos;

    std::vector<trivial_struct> v(16);
    auto proj = [](trivial_struct &s) -> float & { return s.z; };
    auto &&acc = empi::layouts::struct_layout::struct_accessor<trivial_struct, decltype(proj)>(std::move(proj));
    auto view = empi::layouts::contiguous_layout::build(v, acc);

    REQUIRE_FALSE(empi::layouts::is_trivial_view<decltype(view)>);
    auto ptr = empi::layouts::compact(view);
    REQUIRE(empi::details::is_same_template_v<std::unique_ptr<int>, decltype(ptr)>);
}

TEST_CASE("Call compact on a non-contiguous layout and trivial accessor produces a copy", "[compact][layouts]") {
    using namespace Kokkos;

    std::vector<trivial_struct> v(16);
    auto view = empi::layouts::column_layout::build(v, extents<int, 4, 4>{}, 3);

    REQUIRE_FALSE(empi::layouts::is_trivial_view<decltype(view)>);
    auto ptr = empi::layouts::compact(view);
    REQUIRE(empi::details::is_same_template_v<std::unique_ptr<int>, decltype(ptr)>);
}

TEST_CASE("Call compact on a non-contiguous layout and non-trivial accessor produces a copy", "[compact][layouts]") {
    using namespace Kokkos;

    std::vector<trivial_struct> v(16);
    auto proj = [](trivial_struct &s) -> float & { return s.z; };
    auto &&acc = empi::layouts::struct_layout::struct_accessor<trivial_struct, decltype(proj)>(std::move(proj));
    auto view = empi::layouts::column_layout::build(v, extents<int, 4, 4>{}, 3, acc);

    REQUIRE_FALSE(empi::layouts::is_trivial_view<decltype(view)>);
    auto ptr = empi::layouts::compact(view);
    REQUIRE(empi::details::is_same_template_v<std::unique_ptr<int>, decltype(ptr)>);
}

TEST_CASE("Compact block tiled layout", "[mdspan|layouts|compact]") {
    std::vector<int> v(20);
    std::iota(v.begin(), v.end(), 0);

    size_t blocks{4};
    size_t strides{5};
    Kokkos::extents<size_t, 16> ext;
    auto view = empi::layouts::block_layout::build(v, ext, blocks, strides, Kokkos::default_accessor<int>());
    auto ptr = empi::layouts::block_layout::compact(view);
    for(int i = 0; i < view.extent(0); i++) { REQUIRE(ptr.get()[i] == i + (1 * (i / 4))); }
}

// TEST_CASE("Compact block bucket layout", "[mdspan|layouts|compact]"){
// 	std::vector<int> v(32);
// 	std::iota(v.begin(),v.end(),0);

// 	size_t blocks{2,4});
// 	size_t strides{4,4});
// 	constexpr auto size = 24;
// 	Kokkos::extents<size_t,size> ext;
// 	auto view = empi::layouts::block_layout::build(v, ext,
// 														blocks,strides,
// 													   Kokkos::default_accessor<int>());
// 	auto ptr = empi::layouts::block_layout::compact(view);
// 	for (int i = 0; i < view.extent(0); i++) {
// 		// REQUIRE(ptr.get()[i] == i + (1 * (i/4)));
// 		//TODO: not implemented...
// 	}

// }


TEST_CASE("Compact non-contiguous data", "[compact][layouts]") {
    using namespace Kokkos;
    std::vector<int> v(16, 10);
    auto view = empi::layouts::column_layout::build(v, Kokkos::extents<size_t, 4, 4>(), 2);
    auto ptr = empi::layouts::compact(view);

    for(int i = 0; i < 4; i++) { REQUIRE(ptr.get()[i] == 10); }
}

TEST_CASE("Compact a strided layout with contiguous X dim", "[compact][layouts]") {
    int DIM1 = 5;
    int DIM2 = 2;
    int DIM3 = 4;
    std::vector<float> send_array(DIM1 * DIM2 * DIM3);
    std::fill(send_array.begin(), send_array.end(), 0.f);

    Kokkos::mdspan send_mdspan(send_array.data(), DIM3, DIM2, DIM1);
    for(int i = 0; i < send_mdspan.extent(0); i++) {
        for(int k = 0; k < send_mdspan.extent(2); k++) { send_mdspan(i, 0, k) = 1.f; }
        std::cout << std::endl;
    }
    // Print mdspan
    std::cout << "MDSPAN: " << std::endl;
    for(int i = 0; i < send_mdspan.extent(0); i++) {
        for(int j = 0; j < send_mdspan.extent(1); j++) {
            for(int k = 0; k < send_mdspan.extent(2); k++) { std::cout << send_mdspan(i, j, k) << " "; }
            std::cout << std::endl;
        }
    }
    std::cout << "MDSPAN: " << std::endl;


    Kokkos::dextents<std::size_t, 2> ext{DIM3, DIM1};
    std::array<int, 2> strides{DIM1 * DIM2, 1};
    Kokkos::layout_stride::mapping<decltype(ext)> layout(ext, strides);
    Kokkos::mdspan<float, decltype(ext), Kokkos::layout_stride> view{send_array.data(), layout};

    // Print mdspan
    std::cout << "MDSPAN: " << std::endl;
    for(size_t i = 0; i < view.extent(0); i++) {
        for(size_t j = 0; j < view.extent(1); j++) { std::cout << view(i, j) << " "; }
        std::cout << std::endl;
    }

    auto ptr = empi::layouts::compact<empi::details::row_major>(view);
    for(int i = 0; i < DIM1 * DIM3; i++) { std::cout << ptr.get()[i] << " "; }
    std::cout << std::endl;
    for(int i = 0; i < DIM1 * DIM3; i++) { REQUIRE(ptr.get()[i] == 1.f); }
}

TEST_CASE("Compact an indexed layout with non-consecutive indexes", "[compact][layouts]") {
    std::vector<size_t> v(8);
    std::iota(v.begin(), v.end(), 0);
    std::vector<size_t> indices{0, 2, 4, 6};
    auto view = empi::layouts::index_block_layout::build(v, Kokkos::extents<size_t, 4>{}, 1, indices);

    for(int i = 0; i < view.extent(0); i++) { REQUIRE(view(i) == 2 * i); }

    auto ptr = empi::layouts::compact(view);
    for(int i = 0; i < view.extent(0); i++) { REQUIRE(ptr.get()[i] == 2 * i); }
}

TEST_CASE("Compact an indexed layout with some consecutive indexes", "[compact][layouts]") {
    std::vector<size_t> v(8);
    std::iota(v.begin(), v.end(), 0);
    std::vector<size_t> indices{0, 2, 3, 4, 6};
    auto view = empi::layouts::index_block_layout::build(v, Kokkos::extents<size_t, 5>{}, 1, indices);

    REQUIRE(view(0) == 0);
    REQUIRE(view(1) == 2);
    REQUIRE(view(2) == 3);
    REQUIRE(view(3) == 4);
    REQUIRE(view(4) == 6);

    auto ptr = empi::layouts::compact(view);
    REQUIRE(ptr.get()[0] == 0);
    REQUIRE(ptr.get()[1] == 2);
    REQUIRE(ptr.get()[2] == 3);
    REQUIRE(ptr.get()[3] == 4);
    REQUIRE(ptr.get()[4] == 6);
}

TEST_CASE("Compact an indexed layout and create a resulting view with consecutive elements", "[compact][layouts]") {
    constexpr size_t data_size = 8;
    std::vector<size_t> v(data_size);
    std::iota(v.begin(), v.end(), 10);
    std::vector<size_t> indices{0, 1, 2, 3, 4, 5, 6, 7};
    // permute indices
    std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});
    auto view = empi::layouts::index_block_layout::build(v, Kokkos::dextents<size_t, 1>{indices.size()}, 1, indices);

    REQUIRE(view(0) == v[indices[0]]);
    REQUIRE(view(1) == v[indices[1]]);
    REQUIRE(view(2) == v[indices[2]]);
    REQUIRE(view(3) == v[indices[3]]);
    REQUIRE(view(4) == v[indices[4]]);
    REQUIRE(view(5) == v[indices[5]]);
    REQUIRE(view(6) == v[indices[6]]);
    REQUIRE(view(7) == v[indices[7]]);

    auto ptr = empi::layouts::compact(view);
    auto unpack_indices
        = empi::layouts::index_block_layout::get_unpack_indices(view); // REQUIRE(ptr.get()[0] == v[indices[0]]);
    auto res_view = empi::layouts::index_block_layout::build(
        std::span<size_t>(ptr.get(), 8), Kokkos::dextents<size_t, 1>{indices.size()}, 1, unpack_indices);
    REQUIRE(res_view(0) == v[indices[0]]);
    REQUIRE(res_view(1) == v[indices[1]]);
    REQUIRE(res_view(2) == v[indices[2]]);
    REQUIRE(res_view(3) == v[indices[3]]);
    REQUIRE(res_view(4) == v[indices[4]]);
    REQUIRE(res_view(5) == v[indices[5]]);
    REQUIRE(res_view(6) == v[indices[6]]);
    REQUIRE(res_view(7) == v[indices[7]]);
}

TEST_CASE("Compact an indexed layout and create a resulting view with non-consecutive elements", "[compact][layouts]") {
    constexpr size_t data_size = 8;
    std::vector<size_t> v(data_size);
    std::iota(v.begin(), v.end(), 10);
    std::vector<size_t> indices{0, 2, 5, 3, 7};
    // permute indices
    std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});
    const auto view_size = indices.size();
    auto view = empi::layouts::index_block_layout::build(v, Kokkos::dextents<size_t, 1>{view_size}, 1, indices);

    REQUIRE(view.size() == view_size);
    for(size_t i = 0; i < view_size; i++) { REQUIRE(view(i) == v[indices[i]]); }

    auto ptr = empi::layouts::compact(view);
    auto unpack_indices
        = empi::layouts::index_block_layout::get_unpack_indices(view);
    auto res_view = empi::layouts::index_block_layout::build(
        std::span<size_t>(ptr.get(), view_size), Kokkos::dextents<size_t, 1>{view_size}, 1, unpack_indices);
    for(size_t i = 0; i < view_size; i++) { REQUIRE(res_view(i) == v[indices[i]]); }
}