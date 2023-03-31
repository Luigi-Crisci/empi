#ifndef INCLUDE_EMPI_COMPACT
#define INCLUDE_EMPI_COMPACT
#include <memory>

#include <experimental/mdspan>
#include <empi/type_traits.hpp>
#include <empi/defines.hpp>
#include <empi/layouts_traits.hpp>
#include <empi/utils.hpp>


namespace empi::layouts{

/**
	Plain compact strategy for non-contiguous, sparse data 
*/
template<typename T, template<typename , size_t...> typename Extents, typename Layout, typename Accessor, typename idx_type, size_t ...idx>
auto compact(const stdex::mdspan<T,Extents<idx_type,idx...>,Layout,Accessor>& view){
	using element_type = std::remove_cvref_t<typename Accessor::element_type>;
	auto ptr = new element_type[view.size()];
	empi::details::apply(view, [p = ptr](Accessor::reference e) mutable {*p=e; p++;});
	details::conditional_deleter<element_type> del(true);
	std::unique_ptr<element_type, decltype(del)> uptr(std::move(ptr), std::move(del)); 
	return uptr;
}

/**
	If data are contiguous, and we have a trivial accessor we don't have to make any copies
*/
template<typename T, template<typename , size_t...> typename Extents, typename Layout, typename Accessor, typename idx_type, size_t ...idx>
requires (is_trivial_view<Layout, Accessor>)
auto constexpr compact(const stdex::mdspan<T,Extents<idx_type,idx...>,Layout,Accessor>& view){
	using element_type = std::remove_cvref_t<typename Accessor::element_type>;
	
	std::unique_ptr<element_type, details::conditional_deleter<element_type>> uptr(view.data_handle()); 
	return uptr;
}




/**
* TODO: Here we can have some fun
		Extents can be static, enabling compile-time loop unrolling/vectorization
		Layouts have predictable structure which can be exploit for optimizing data movements
		(e.g. a tiled layout have contiguos elements on dim 0, and potentally on dim 1 also)\
		Said so, we can potentially specialize the compact function in several flavours  
*/


}

namespace empi::details{
template<typename T, typename Extents, typename Layout, typename Accessor>
static constexpr inline auto get_underlying_pointer(const stdex::mdspan<T, Extents,Layout,Accessor>& buf, 
													bool compact = false){
  if (compact){
	// TODO: temporary workaround before refactoring compact functions into layout's classes
	if constexpr (std::is_same_v<Layout, layouts::block_layout>)
		return empi::layouts::block_layout::compact(buf);
	else
		return empi::layouts::compact(buf);
  } else {
	using element_type = std::remove_cvref_t<typename Accessor::element_type>;
	return std::unique_ptr<element_type,details::conditional_deleter<element_type>>(buf.data_handle());
  }
}

}

#endif /* INCLUDE_EMPI_COMPACT */
