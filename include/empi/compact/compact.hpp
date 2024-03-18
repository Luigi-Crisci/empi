#ifndef INCLUDE_EMPI_COMPACT
#define INCLUDE_EMPI_COMPACT
#include <memory>

#include <empi/defines.hpp>
#include <empi/layouts_traits.hpp>
#include <empi/type_traits.hpp>
#include <empi/utils.hpp>
#include <mdspan/mdspan.hpp>

#include "common_compact.hpp"
#include "compact_layout_stride.hpp"


namespace empi::details {
template<typename T, typename Extents, typename Layout, typename Accessor>
static constexpr inline auto get_underlying_pointer(
    const Kokkos::mdspan<T, Extents, Layout, Accessor> &buf, bool compact = false) {
    if(compact) {
        // TODO: temporary workaround before refactoring compact functions into layout's classes
        if constexpr(details::is_block_layout<Layout>)
            return empi::layouts::block_layout::compact(buf);
        else
            return empi::layouts::compact(buf);
    } else {
        using element_type = std::remove_cvref_t<typename Accessor::element_type>;
        if constexpr(layouts::has_trivial_accessor<Accessor>) // Workaround because otherwise unique_ptr complains
                                                              // about mismatching types with struct accessors
            return std::unique_ptr<element_type, details::conditional_deleter<element_type>>(buf.data_handle());
    }
    __builtin_unreachable();
}

} // namespace empi::details

#endif /* INCLUDE_EMPI_COMPACT */
