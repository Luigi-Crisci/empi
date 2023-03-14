#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <empi/empi.hpp>
#include <type_traits>

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