#include <catch2/catch_test_macros.hpp>
#include <empi/empi.hpp>
#include <empi/compact.hpp>

#include "utils.hpp"

namespace stdex = std::experimental;

TEST_CASE("Call compact on a trivial layout and accessor does not produce copies", "[compact][layouts]"){
	using namespace std::experimental;
	std::vector<int> v(16);
	auto view = empi::layouts::contiguous_layout::build(v);
	REQUIRE(empi::layouts::is_trivial_view<decltype(view)::layout_type, decltype(view)::accessor_type>);
	auto ptr = empi::layouts::compact(view);
		
	REQUIRE(empi::details::is_same_template_v<empi::details::conditional_deleter<int>,decltype(ptr)::deleter_type>);
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

TEST_CASE("Compact block tiled layout", "[mdspan|layouts|compact]"){
	std::vector<int> v(20);
	std::iota(v.begin(),v.end(),0);

	size_t blocks{4};
	size_t strides{5};
	stdex::extents<size_t,16> ext;
	auto view = empi::layouts::block_layout::build(v, ext, 
														blocks,strides,
													    std::experimental::default_accessor<int>());
	auto ptr = empi::layouts::block_layout::compact(view);
	for (int i = 0; i < view.extent(0); i++) {
		REQUIRE(ptr.get()[i] == i + (1 * (i/4)));
	}
	
}

// TEST_CASE("Compact block bucket layout", "[mdspan|layouts|compact]"){
// 	std::vector<int> v(32);
// 	std::iota(v.begin(),v.end(),0);

// 	size_t blocks{2,4});
// 	size_t strides{4,4});
// 	constexpr auto size = 24;
// 	stdex::extents<size_t,size> ext;
// 	auto view = empi::layouts::block_layout::build(v, ext, 
// 														blocks,strides,
// 													    std::experimental::default_accessor<int>());
// 	auto ptr = empi::layouts::block_layout::compact(view);
// 	for (int i = 0; i < view.extent(0); i++) {
// 		// REQUIRE(ptr.get()[i] == i + (1 * (i/4)));
// 		//TODO: not implemented...
// 	}
	
// }



TEST_CASE("Compact non-contiguous data", "[compact][layouts]"){
	using namespace std::experimental;
	std::vector<int> v(16, 10);
	auto view = empi::layouts::column_layout::build(v, stdex::extents<size_t, 4,4>(), 2);
	auto ptr = empi::layouts::compact(view);
	
	for (int i = 0; i < 4; i++) {
		REQUIRE(ptr.get()[i] == 10);
	}

}