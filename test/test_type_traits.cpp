#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <type_traits>

#include <empi/defines.hpp>
#include <empi/empi.hpp>
#include <empi/layouts_traits.hpp>
#include <empi/type_traits.hpp>

namespace stdex = Kokkos;

TEST_CASE("is_mdspan", "[type_traits]") {
    using namespace Kokkos;
    using extent_type = dextents<int, 1>;

    REQUIRE_FALSE(empi::details::is_mdspan<int>);
    REQUIRE_FALSE(empi::details::is_mdspan<std::vector<int>>);

    REQUIRE(empi::details::is_mdspan<mdspan<int, extent_type>>);
    REQUIRE(empi::details::is_mdspan<mdspan<float, extent_type>>);
    REQUIRE(empi::details::is_mdspan<mdspan<int, extents<int, 1>>>);
    REQUIRE(empi::details::is_mdspan<mdspan<float, extents<int, 1>>>);
}


TEST_CASE("is_contiguous_layout", "[type_traits][layouts]") {
    using namespace Kokkos;

    REQUIRE(empi::layouts::is_contiguous_layout<Kokkos::layout_right>);
    REQUIRE(empi::layouts::is_contiguous_layout<Kokkos::layout_left>);
    REQUIRE(empi::layouts::is_contiguous_layout<empi::layouts::contiguous_layout::contiguous_layout_impl>);

    REQUIRE_FALSE(empi::layouts::is_contiguous_layout<Kokkos::layout_stride>);
    REQUIRE_FALSE(empi::layouts::is_contiguous_layout<int>);
}

TEST_CASE("has_trivial_accessor", "[type_traits][layouts]") {
    using namespace Kokkos;
    struct S {
        int x;
        int y;
        float z;
    };

    REQUIRE(empi::layouts::has_trivial_accessor<Kokkos::default_accessor<int>>);
    REQUIRE(empi::layouts::has_trivial_accessor<empi::layouts::struct_layout::struct_accessor<int>>);
    REQUIRE(empi::layouts::has_trivial_accessor<empi::layouts::struct_layout::struct_accessor<S>>);

    REQUIRE_FALSE(empi::layouts::has_trivial_accessor<
        empi::layouts::struct_layout::struct_accessor<S, decltype([](S &s) -> int & { return s.x; })>>);
    REQUIRE_FALSE(empi::layouts::has_trivial_accessor<empi::layouts::struct_layout::struct_accessor<S,
            decltype([](S &s) -> std::tuple<int, float> { return {s.x, s.z}; })>>);
}