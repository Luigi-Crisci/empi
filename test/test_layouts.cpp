#include <catch2/catch_test_macros.hpp>

#include <cstdint>

#include <empi/empi.hpp>

template<typename T, class Extents, class Layout>
requires (Extents::rank() == 1)
void print(const empi::stdex::mdspan<T, Extents, Layout>& view){
	std::cout <<"\t\t";
	for(int i=0; i < view.extent(0); ++i){
			std::cout << view[i] << ", ";
	}
}

template<typename T, class Extents>
requires (Extents::rank() == 2)
void print(const empi::stdex::mdspan<T, Extents>& view){
	for(int i=0; i < view.extent(0); ++i){
		if(i != 0) std::cout << "\n";
			std::cout <<"\t";
		for(int j = 0; j < view.extent(1); ++j){
			std::cout << view(i,j) << ", ";
		}
	}
}

TEST_CASE("Create column mdspan", "[column_layout]") {
	std::vector<int> v(10);
	std::iota(v.begin(),v.end(),0);

	empi::stdex::extents<size_t,2,5> ext;
	empi::stdex::mdspan<int, decltype(ext)> tmp(v.data(),ext); 
	auto view = empi::layouts::column_layout::build(v, ext, 0);

	REQUIRE(decltype(view)::rank() == 1);
	REQUIRE(view.extent(0) == 2);

	for(int i = 1, j = 0; i < 10; i+=2,j++)
		REQUIRE(v[i] == view[j]);
}


TEST_CASE("Check extents type traits", "[mdspan]") {
	empi::stdex::extents<int, 1,2,3,4,5> ex;
	using trimmed_extent = decltype(empi::remove_last(ex));
	REQUIRE(std::is_same_v<trimmed_extent,empi::stdex::extents<int, 1,2,3,4>>);
}

struct S{
	int x;
	int y;
};

TEST_CASE("Create struct mdspan", "[mdspan|struct_layout]") {
	std::vector<S> tmp(10);
	int count = 0;
	std::transform(tmp.begin(), tmp.end(), tmp.begin(),[&count](S& s){s.x=count++; s.y=count++;return s;});
	auto proj = [](S& s) -> int& {return s.y;};
    auto ext = empi::stdex::extents(10);
    empi::stdex::mdspan view{tmp.data(), empi::stdex::layout_right::mapping{ext}, empi::layouts::struct_layout::struct_accessor<S,decltype(proj)>(std::move(proj))};

	for(int i = 0; i < 10; i+=1){
		REQUIRE(view[i] == i*2+1);
	}
}

