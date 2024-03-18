#include <catch2/catch_test_macros.hpp>
#include <empi/compact/compact.hpp>
#include <empi/empi.hpp>

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

    REQUIRE_FALSE(
        empi::layouts::is_trivial_view<decltype(view)>);
    auto ptr = empi::layouts::compact(view);
    REQUIRE(empi::details::is_same_template_v<std::unique_ptr<int>, decltype(ptr)>);
}

TEST_CASE("Call compact on a non-contiguous layout and trivial accessor produces a copy", "[compact][layouts]") {
    using namespace Kokkos;

    std::vector<trivial_struct> v(16);
    auto view = empi::layouts::column_layout::build(v, extents<int, 4, 4>{}, 3);

    REQUIRE_FALSE(
        empi::layouts::is_trivial_view<decltype(view)>);
    auto ptr = empi::layouts::compact(view);
    REQUIRE(empi::details::is_same_template_v<std::unique_ptr<int>, decltype(ptr)>);
}

TEST_CASE("Call compact on a non-contiguous layout and non-trivial accessor produces a copy", "[compact][layouts]") {
    using namespace Kokkos;

    std::vector<trivial_struct> v(16);
    auto proj = [](trivial_struct &s) -> float & { return s.z; };
    auto &&acc = empi::layouts::struct_layout::struct_accessor<trivial_struct, decltype(proj)>(std::move(proj));
    auto view = empi::layouts::column_layout::build(v, extents<int, 4, 4>{}, 3, acc);

    REQUIRE_FALSE(
        empi::layouts::is_trivial_view<decltype(view)>);
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

TEST_CASE("Compact a strided layout with contiguous X dim", "[compact][layouts]"){
        int DIM1 = 5;
        int DIM2 = 2;
        int DIM3 = 4;
        std::vector<float> send_array(DIM1 * DIM2 * DIM3);
        std::fill(send_array.begin(), send_array.end(), 0.f);
        
        Kokkos::mdspan send_mdspan(send_array.data(), DIM3, DIM2, DIM1);
        for (int i = 0; i < send_mdspan.extent(0); i++){
                for (int k = 0; k < send_mdspan.extent(2); k++){
                    send_mdspan(i,0,k) = 1.f;
                }
                std::cout << std::endl;
        }
        //Print mdspan
        std::cout << "MDSPAN: " << std::endl;
        for (int i = 0; i < send_mdspan.extent(0); i++){
            for (int j = 0; j < send_mdspan.extent(1); j++){
                for (int k = 0; k < send_mdspan.extent(2); k++){
                    std::cout << send_mdspan(i,j,k) << " ";
                }
                std::cout << std::endl;
            }
        }
        std::cout << "MDSPAN: " << std::endl;




        Kokkos::dextents<std::size_t, 2> ext{DIM3, DIM1};
        std::array<int, 2> strides{DIM1 * DIM2, 1};
        Kokkos::layout_stride::mapping<decltype(ext)> layout(ext, strides);
        Kokkos::mdspan<float, decltype(ext), Kokkos::layout_stride> view{send_array.data(), layout};

        //Print mdspan
        std::cout << "MDSPAN: " << std::endl;
        for(size_t i = 0; i < view.extent(0); i++) {
            for(size_t j = 0; j < view.extent(1); j++) {
                std::cout << view(i, j) << " ";
            }
            std::cout << std::endl;
        }        

        auto ptr = empi::layouts::compact<empi::details::row_major>(view);
        for (int i = 0; i < DIM1 * DIM3; i++) {
            std::cout << ptr.get()[i] << " ";
        }
        std::cout << std::endl;
        for(int i = 0; i < DIM1 * DIM3; i++) { REQUIRE(ptr.get()[i] == 1.f); }
}