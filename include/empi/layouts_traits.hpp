#ifndef INCLUDE_EMPI_LAYOUTS_TRAITS
#define INCLUDE_EMPI_LAYOUTS_TRAITS
#include <type_traits>
#include <empi/layouts.hpp>
#include <empi/type_traits.hpp>
#include <empi/datatype.hpp>

namespace empi::layouts{

template<typename Layout>
concept is_contiguous_layout = std::is_same_v<Layout, contiguous_layout::contiguous_layout_impl> || 
							   std::is_same_v<Layout, stdex::layout_right> || 
							   std::is_same_v<Layout, stdex::layout_left>;

template<typename AccessPolicy>
concept is_trivial_accessor = std::is_same_v<details::remove_all_t<typename AccessPolicy::data_handle_type>, 
			       							 details::remove_all_t<typename AccessPolicy::element_type>>;

template<typename Layout, typename AccessPolicy>
concept is_trivial_view = is_contiguous_layout<Layout> && is_trivial_accessor<AccessPolicy>;


}

#endif /* INCLUDE_EMPI_LAYOUTS_TRAITS */
