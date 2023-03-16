#include <catch2/catch_test_macros.hpp>
#include <empi/empi.hpp>
#include <empi/compact.hpp>

#include "utils.hpp"

TEST_CASE("Call compact on a trivial layout and accessor does not produce copies", "[compact][layouts]"){
	using namespace std::experimental;
	std::vector<int> v(16);
	auto view = empi::layouts::contiguous_layout::build(v);
	REQUIRE(empi::layouts::is_trivial_view<decltype(view)::layout_type, decltype(view)::accessor_type>);
	auto ptr = empi::layouts::compact(view);
	REQUIRE(empi::details::is_same_template_v<empi::details::pointer_wrapper<int>, decltype(ptr)>);
}

TEST_CASE("Call compact on a contiguous layout and non-trivial accessor produces a copy", "[compact][layouts]"){
	using namespace std::experimental;

	std::vector<trivial_struct> v(16);
	auto proj = [](trivial_struct& s) -> float& {return s.z;};
	auto&& acc = empi::layouts::struct_layout::struct_accessor<trivial_struct,decltype(proj)>(std::move(proj));
	auto view = empi::layouts::contiguous_layout::build(v,acc);
	
	REQUIRE_FALSE(empi::layouts::is_trivial_view<typename decltype(view)::layout_type, typename decltype(view)::accessor_type>);
	auto ptr = empi::layouts::compact(view);
	REQUIRE(empi::details::is_same_template_v<std::unique_ptr<int>, decltype(ptr)>);
}

TEST_CASE("Call compact on a non-contiguous layout and trivial accessor produces a copy", "[compact][layouts]"){
	using namespace std::experimental;

	std::vector<trivial_struct> v(16);
	auto view = empi::layouts::column_layout::build(v, extents<int, 4,4>{}, 3);
	
	REQUIRE_FALSE(empi::layouts::is_trivial_view<typename decltype(view)::layout_type, typename decltype(view)::accessor_type>);
	auto ptr = empi::layouts::compact(view);
	REQUIRE(empi::details::is_same_template_v<std::unique_ptr<int>, decltype(ptr)>);
}

TEST_CASE("Call compact on a non-contiguous layout and non-trivial accessor produces a copy", "[compact][layouts]"){
	using namespace std::experimental;

	std::vector<trivial_struct> v(16);
	auto proj = [](trivial_struct& s) -> float& {return s.z;};
	auto&& acc = empi::layouts::struct_layout::struct_accessor<trivial_struct,decltype(proj)>(std::move(proj));
	auto view = empi::layouts::column_layout::build(v,extents<int,4,4>{}, 3, acc);
	
	REQUIRE_FALSE(empi::layouts::is_trivial_view<typename decltype(view)::layout_type, typename decltype(view)::accessor_type>);
	auto ptr = empi::layouts::compact(view);
	REQUIRE(empi::details::is_same_template_v<std::unique_ptr<int>, decltype(ptr)>);
}
