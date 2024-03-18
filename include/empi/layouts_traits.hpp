#ifndef INCLUDE_EMPI_LAYOUTS_TRAITS
#define INCLUDE_EMPI_LAYOUTS_TRAITS

#include <empi/layouts.hpp>
#include <empi/type_traits.hpp>
#include <type_traits>

namespace empi::layouts {

template<typename Layout>
concept is_contiguous_layout = std::is_same_v<Layout, contiguous_layout::contiguous_layout_impl>
    || std::is_same_v<Layout, Kokkos::layout_right> || std::is_same_v<Layout, Kokkos::layout_left>;



template<typename AccessPolicy>
concept has_trivial_accessor = std::is_same_v<details::remove_all_t<typename AccessPolicy::data_handle_type>,
    details::remove_all_t<typename AccessPolicy::element_type>>;
template<typename Layout>
concept is_block_layout = std::is_same_v<Layout, layouts::block_layout>;

template<typename Mdspan>
concept is_trivial_view = details::is_mdspan_v<Mdspan> && is_contiguous_layout<typename Mdspan::layout_type> && has_trivial_accessor<typename Mdspan::access_policy>;



} // namespace empi::layouts

#endif /* INCLUDE_EMPI_LAYOUTS_TRAITS */
