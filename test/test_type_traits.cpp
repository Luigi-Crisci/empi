#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <type_traits>

#include <empi/empi.hpp>
#include <empi/type_traits.hpp>
#include <empi/layouts_traits.hpp>
#include <empi/defines.hpp>

namespace stdex = std::experimental;

TEST_CASE("is_mdspan", "[type_traits]") {
	using namespace std::experimental;
	using extent_type = dextents<int, 1>;
	
	REQUIRE_FALSE(empi::details::is_mdspan<int>);
	REQUIRE_FALSE(empi::details::is_mdspan<std::vector<int>>);
	
	REQUIRE(empi::details::is_mdspan<mdspan<int, extent_type>>);
	REQUIRE(empi::details::is_mdspan<mdspan<float, extent_type>>);
	REQUIRE(empi::details::is_mdspan<mdspan<int, extents<int, 1>>>);
	REQUIRE(empi::details::is_mdspan<mdspan<float, extents<int, 1>>>);
}


TEST_CASE("is_contiguous_layout", "[type_traits][layouts]") {
	using namespace std::experimental;

	REQUIRE(empi::layouts::is_contiguous_layout<stdex::layout_right>);	
	REQUIRE(empi::layouts::is_contiguous_layout<stdex::layout_left>);	
	REQUIRE(empi::layouts::is_contiguous_layout<empi::layouts::contiguous_layout::contiguous_layout_impl>);	

	REQUIRE_FALSE(empi::layouts::is_contiguous_layout<stdex::layout_stride>);
	REQUIRE_FALSE(empi::layouts::is_contiguous_layout<int>);
}

TEST_CASE("is_trivial_accessor", "[type_traits][layouts]") {
	using namespace std::experimental;
	struct S{
		int x;
		int y;
		float z;
	};

	REQUIRE(empi::layouts::is_trivial_accessor<stdex::default_accessor<int>>);	
	REQUIRE(empi::layouts::is_trivial_accessor<empi::layouts::struct_layout::struct_accessor<int>>);	
	REQUIRE(empi::layouts::is_trivial_accessor<empi::layouts::struct_layout::struct_accessor<S>>);	
	
	REQUIRE_FALSE(empi::layouts::is_trivial_accessor<empi::layouts::struct_layout::struct_accessor<S,
																					decltype([](S&s)->int&{return s.x;})>>);
	REQUIRE_FALSE(empi::layouts::is_trivial_accessor<empi::layouts::struct_layout::struct_accessor<S,
																					decltype([](S&s)->std::tuple<int,float> {return {s.x,s.z};})>>);
}