#ifndef EMPI_COMMON_COMPACT
#define EMPI_COMMON_COMPACT

#include <empi/defines.hpp>
#include <empi/layouts_traits.hpp>
#include <empi/type_traits.hpp>
#include <empi/utils.hpp>
#include <mdspan/mdspan.hpp>

namespace empi::layouts {

using namespace empi::details;

// Plain compact strategy for non-contiguous, sparse data
template<typename T, template<typename, size_t...> typename Extents, typename Layout, typename Accessor,
    typename IdxType, size_t... Idx>
constexpr auto non_contiguous_layout_compact_impl(const Kokkos::mdspan<T, Extents<IdxType, Idx...>, Layout, Accessor> &view) {
    using element_type = std::remove_cvref_t<typename Accessor::element_type>;
    auto ptr = new element_type[view.size()];
    empi::details::apply(view, [p = ptr](typename Accessor::reference e) mutable {
        *p = e;
        p++;
    });
    conditional_deleter<element_type> del(true);
    std::unique_ptr<element_type, decltype(del)> uptr(std::move(ptr), std::move(del));
    return uptr;
}

template<typename T>
requires Mdspan<T>
constexpr auto compact(T&& view) {
    return non_contiguous_layout_compact_impl(std::forward<T>(view));
}

/**
    If data are contiguous, and we have a trivial accessor we don't have to make any copies
*/
template<typename T, template<typename, size_t...> typename Extents, typename Layout, typename Accessor,
    typename IdxType, size_t... Idx>
auto constexpr contiguous_layout_compact_impl(const Kokkos::mdspan<T, Extents<IdxType, Idx...>, Layout, Accessor> &view) {
    using element_type = std::remove_cvref_t<typename Accessor::element_type>;
    std::unique_ptr<element_type, conditional_deleter<element_type>> uptr(view.data_handle());
    return uptr;
}

template<typename T>
requires Mdspan<T> && is_trivial_view<T>
constexpr auto compact(T&& view) {
    return contiguous_layout_compact_impl(std::forward<T>(view));
}

/**
* TODO: Here we can have some fun
        Extents can be static, enabling compile-time loop unrolling/vectorization
        Layouts have predictable structure which can be exploit for optimizing data movements
        (e.g. a tiled layout have contiguos elements on dim 0, and potentally on dim 1 also)\
        Said so, we can potentially specialize the compact function in several flavours
*/


} // namespace empi::layouts


#endif /* EMPI_COMMON_COMPACT */
