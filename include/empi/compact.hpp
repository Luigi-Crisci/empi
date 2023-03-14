#ifndef INCLUDE_EMPI_COMPACT
#define INCLUDE_EMPI_COMPACT

#include <memory>
#include <experimental/mdspan>
#include <empi/type_traits.hpp>
#include <empi/datatype.hpp>
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
	std::unique_ptr<element_type> uptr(std::move(ptr)); 
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

#endif /* INCLUDE_EMPI_COMPACT */
