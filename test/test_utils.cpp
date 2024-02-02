#include <catch2/catch_test_macros.hpp>
#include <empi/layouts_traits.hpp>
#include <empi/utils.hpp>


TEST_CASE("Conditional deleter to non-owning memory does not free memory", "[conditional_deleter]") {
    int *v = new int[10];
    // Create and deallocate unique_ptr
    { std::unique_ptr<int, empi::details::conditional_deleter<int>> tmp(v); }
    REQUIRE(v != nullptr);
    delete[] v;
}

TEST_CASE("Conditional deleter to owning memory frees memory", "[conditional_deleter]") {
    int *v = new int[10];
    empi::details::conditional_deleter<int> del(true);
    // Create and deallocate unique_ptr
    { std::unique_ptr<int, decltype(del)>(v, del); }
    // magic check here
}

TEST_CASE("Mdspan can be modified through ptr reference", "[mdspan]") {
    std::vector<int> v(10);
    auto view = empi::layouts::contiguous_layout::build(v);
    auto ptr = view.data_handle();
    ptr[0] = 6;
    REQUIRE(ptr[0] == 6);
    REQUIRE(view[0] == 6);
}