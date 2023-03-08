#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <empi/empi.hpp>
#include <type_traits>

TEST_CASE("is_mdspan", "[type_traits]") {
	using namespace std::experimental;
	using extent_type = dextents<int, 1>;
	
	REQUIRE_FALSE(std::is_same_v<mdspan<int, extent_type>, int>);
	REQUIRE_FALSE(std::is_same_v<mdspan<int, extent_type>, std::vector<int>>);
	
	REQUIRE(std::is_same_v<mdspan<int, extent_type>, mdspan<int, extent_type>>);
	REQUIRE(std::is_same_v<mdspan<int, extent_type>, mdspan<float, extent_type>>);
	REQUIRE(std::is_same_v<mdspan<int, extent_type>, mdspan<int, extents<int, 1>>>);
	REQUIRE(std::is_same_v<mdspan<int, extent_type>, mdspan<float, extents<int, 1>>>);


}