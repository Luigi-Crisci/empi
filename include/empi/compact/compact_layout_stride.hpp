#ifndef EMPI_COMPACT_LAYOUT_STRIDE_HPP
#define EMPI_COMPACT_LAYOUT_STRIDE_HPP

#include <empi/defines.hpp>
#include <empi/layouts_traits.hpp>
#include <empi/type_traits.hpp>
#include <empi/utils.hpp>
#include <mdspan/mdspan.hpp>

namespace empi::details{

    template<size_t Size>
    struct unit_stride_struct {
        bool unit_strides[Size] = {false};
        bool is_contiguous = true;
    };

    template<size_t Size, typename Range>
    requires std::ranges::forward_range<Range>
    auto check_unit_strides(const Range& range) {
        unit_stride_struct<Size> result;
        auto it = std::begin(range);
        auto end = std::end(range);
        for (size_t i = 0; i < Size && it != end; i++){
            if (*it == 1){
                result.unit_strides[i] = true;
            }
            else{
                result.is_contiguous = false;
            }
            it++;
        }
        return result;
    }
}
namespace empi::layouts {


using namespace empi::details;

template<order DataOrdering = row_major, typename T, template<typename, size_t...> typename Extents, typename Layout, typename Accessor,
    typename IdxType, size_t... Idx>
constexpr auto layout_stride_compact_impl2_d(const Kokkos::mdspan<T, Extents<IdxType, Idx...>, Layout, Accessor> &view) {
    using view_type = Kokkos::mdspan<T, Extents<IdxType, Idx...>, Layout, Accessor>;
    using extent_type = std::remove_cvref_t<Extents<IdxType, Idx...>>;
    using element_type = std::remove_cvref_t<typename Accessor::element_type>;
    const Kokkos::layout_stride::mapping<extent_type>& layout = view.mapping();
    const auto& strides = layout.strides();

    //check if we have a dimension with stride 1
    auto unit_stride = check_unit_strides<2>(strides);
    if (unit_stride.is_contiguous){
        return contiguous_layout_compact_impl(view);
    }

    bool has_unit_stride = false;
    if constexpr (DataOrdering == row_major){
        has_unit_stride = (strides[1] == 1);
    }
    else if constexpr (DataOrdering == column_major){
        has_unit_stride = (strides[0] == 1);
    }

    if (!has_unit_stride){
        return non_contiguous_layout_compact_impl(std::move(view));
    }

    //The layout is contiguous only on dim 0
    auto ptr = new element_type[view.size()];
    auto base = ptr;
    for (int i = 0; i < view.extent(0); i++){
        std::copy(&view(i,0), &view(i, view.extent(1)), ptr);
        ptr += view.extent(1);
    }

    conditional_deleter<element_type> del(true);
    std::unique_ptr<element_type, decltype(del)> uptr(std::move(base), std::move(del));
    return uptr;
}

template<order DataOrdering, typename T>
requires Mdspan<T> && std::is_same_v<typename std::remove_cvref_t<T>::layout_type, Kokkos::layout_stride>
constexpr auto compact(T&& view) {
    using extent_type = typename std::remove_cvref_t<T>::extents_type;
    static_assert(extent_type::rank() == 2, "Only 2D mdspan are supported");
    
    if constexpr (extent_type::rank() == 2){
        return layout_stride_compact_impl2_d<DataOrdering>(std::forward<T>(view));
}
}


}

#endif /* EMPI_COMPACT_LAYOUT_STRIDE_HPP */