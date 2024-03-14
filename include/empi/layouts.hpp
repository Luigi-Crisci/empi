#ifndef INCLUDE_EMPI_LAYOUTS
#define INCLUDE_EMPI_LAYOUTS

#include <cassert>
#include <iostream>
#include <memory>
#include <utility>

#include <empi/datatype.hpp>
#include <empi/defines.hpp>
#include <vector>


namespace empi::details {

enum order { row_major = 0, column_major = 1 };

template<typename Extents, typename Indices>
static constexpr Indices linear_index_impl(Indices idx, Extents ext, auto dims) {
    auto local_idx = idx;
    for(auto i = 0; i < dims; i++) { local_idx *= ext.extent(i); }
    return local_idx;
}

template<order Order = row_major, typename Extents, typename IndexType = typename Extents::index_type,
    typename... Indices>
static constexpr IndexType linear_index(Extents ext, Indices... idx) {
    assert(sizeof...(idx) == Extents::rank() && "Number of indices must match the rank of the extents");
    constexpr auto rank = Extents::rank();
    if constexpr(Order == row_major) {
        auto i = rank - 1;
        return (linear_index_impl(idx, ext, i--) + ...);
    }
    if constexpr(Order == column_major) {
        auto i = 0;
        return (linear_index_impl(idx, ext, i++) + ...);
    }
    __builtin_unreachable();
}

template<order Order = row_major, typename Extents, typename IndexType = typename Extents::index_type,
    typename... Indices, size_t N>
static constexpr IndexType linear_index(Extents ext, const std::array<IndexType, N>& idx) {
    assert(idx.size() == Extents::rank() && "Number of indices must match the rank of the extents");
    constexpr auto rank = Extents::rank();
    std::remove_cv_t<IndexType> linear_index{0};
    auto i = Order == row_major ? rank - 1 : 0;
    auto op = [&i]() { return (Order == row_major) ? i-- : i++; };
    std::flush(std::cout);
    for(int j = 0; j < idx.size(); j++) {
        std::flush(std::cout);
        linear_index += linear_index_impl(idx[j], ext, op()); 
    }
    return linear_index; 
}


} // namespace empi::details

namespace empi::layouts {

struct column_layout {
    // Hardwritten for 2D layouts, will change later...
    template<typename Range, template<typename, size_t...> typename Extents, typename K, size_t... Idx>
        requires std::ranges::forward_range<Range>
    [[nodiscard]] static auto build(Range &&view, Extents<K, Idx...> extents, size_t col) {
        return column_layout::build(view, extents, col, Kokkos::default_accessor<std::ranges::range_value_t<Range>>());
    }

    // Hardwritten for 2D layouts, will change later...
    template<typename Range, template<typename, size_t...> typename Extents, typename K, size_t... Idx,
        typename Accessor>
        requires std::ranges::forward_range<Range>
    [[nodiscard]] static auto build(Range &&view, Extents<K, Idx...> extents, size_t col, const Accessor &acc) {
        static_assert(Extents<K, Idx...>::rank() == 2);
        assert(col < extents.extent(1));
        using extent_type = decltype(details::remove_last(extents));

        extent_type new_extents(extents.extent(0));
        std::array stride{extents.extent(1)};
        column_layout_impl::mapping<extent_type> column_map(new_extents, stride);

        using T = std::ranges::range_value_t<decltype(view)>;

        return Kokkos::mdspan<T, extent_type, column_layout_impl, Accessor>(
            std::ranges::data(view) + col, column_map, acc);
    }

    struct column_layout_impl {
        template<typename Extents>
        struct mapping : Kokkos::layout_stride::mapping<Extents> {
            using base = Kokkos::layout_stride::mapping<Extents>;
            using base::base;
        };
    };
};

struct contiguous_layout {
    // Wraps data into plain, 1D mdspan.
    // Used for implicit conversions
    [[nodiscard]] static auto build(std::ranges::forward_range auto &&view) {
        using extent_type = Kokkos::extents<std::size_t, Kokkos::dynamic_extent>;
        extent_type extents(std::ranges::size(view));
        using T = std::ranges::range_value_t<decltype(view)>;
        return std::move(Kokkos::mdspan<T, extent_type, contiguous_layout_impl>(std::ranges::data(view), extents));
    }

    template<typename Accessor>
    [[nodiscard]] static auto build(std::ranges::forward_range auto &&view, const Accessor &acc) {
        using extent_type = Kokkos::dextents<std::size_t, 1>;
        extent_type extents(std::ranges::size(view));
        using T = std::ranges::range_value_t<decltype(view)>;
        return std::move(Kokkos::mdspan<T, extent_type, contiguous_layout_impl, Accessor>(
            std::ranges::data(view), contiguous_layout_impl::mapping<extent_type>(extents), acc));
    }

    struct contiguous_layout_impl {
        template<typename Extents>
        struct mapping : Kokkos::layout_right::mapping<Extents> {
            using base = Kokkos::layout_right::mapping<Extents>;
            using base::base;
        };
    };
};

struct struct_layout {
    template<typename ElementType>
    static constexpr auto default_access = [](ElementType &value) -> ElementType & { return value; };

    template<typename ElementType, typename Callable = std::remove_cv_t<decltype(default_access<ElementType>)>>
    struct struct_accessor {
        using offset_policy = Kokkos::default_accessor<ElementType>;
        using element_type = typename details::function_traits<Callable>::result_type;
        using reference = std::conditional_t<details::is_tuple<element_type>, element_type, element_type &>;
        using data_handle_type = ElementType *;

        constexpr struct_accessor() : m_proj(default_access<ElementType>){};
        explicit struct_accessor(Callable &&c) : m_proj(c) {}

        constexpr reference access(data_handle_type p, size_t i) const noexcept {
            if constexpr(std::is_rvalue_reference_v<reference>) {
                return (m_proj(std::move(p[i])));
            } else {
                return m_proj(p[i]);
            }
        }

      private:
        Callable m_proj;
    };
};

template<typename S, typename Callable>
auto make_struct_accessor(Callable &&c) {
    return struct_layout::struct_accessor<S, Callable>(std::forward<Callable>(c));
}

struct submatrix_layout {
    template<typename T, template<typename, size_t...> typename Extents, typename Layout, typename Accessor,
        typename IdxType, size_t... Idx>
        requires(sizeof...(Idx) == 2)
    [[nodiscard]] static auto build(const Kokkos::mdspan<T, Extents<IdxType, Idx...>, Layout, Accessor> &view,
        std::size_t row_size, std::size_t col_size, std::size_t tile_num, size_t stride = 1) {
        const size_t num_cols = view.extent(1);
        const size_t row_tile_pos = (tile_num / (num_cols / col_size)) * row_size;
        const size_t col_tile_pos = (tile_num % (num_cols / col_size)) * col_size;

        return Kokkos::submdspan(view, Kokkos::strided_slice<size_t, size_t, size_t>{row_tile_pos, row_size, stride},
            Kokkos::strided_slice<size_t, size_t, size_t>{col_tile_pos, col_size, stride});
    }

    // One-dimensional compact
    template<typename T, template<typename, size_t...> typename Extents, typename Layout, typename Accessor,
        typename IdxType, size_t... Idx>
        requires(sizeof...(Idx) == 2)
    static constexpr auto compact(const Kokkos::mdspan<T, Extents<IdxType, Idx...>, Layout, Accessor> &view) {
        using element_type = std::remove_cvref_t<typename Accessor::element_type>;
        auto ptr = new element_type[view.size()];
        auto base = ptr;

        for(size_t row = 0; row < view.extent(0); ++row) {
            std::copy(&view(row, 0), &view(row, view.extent(1)), ptr);
            ptr += view.extent(1);
        }

        details::conditional_deleter<element_type> del(true);
        return std::unique_ptr<element_type, decltype(del)>(std::move(base), del);
    }
};

struct tiled_layout {
    // Hardwritten for 2D layouts, will change later...
    template<typename Range>
        requires std::ranges::forward_range<Range>
    [[nodiscard]] static constexpr auto build(Range &&view, const size_t row, const size_t col) {
        return tiled_layout::build(
            view, Kokkos::dextents<size_t, 2>(row, col), Kokkos::default_accessor<std::ranges::range_value_t<Range>>());
    }

    // Hardwritten for 2D layouts, will change later...
    template<template<typename, size_t...> typename Extents, typename T, size_t... Idx, typename Accessor>
    [[nodiscard]] static constexpr auto build(
        std::ranges::forward_range auto &&view, Extents<T, Idx...> extents, const Accessor &acc) {
        // TODO: Check sizes against view
        // TODO: Tiled layout should work on every dimension
        using extent_type = Extents<T, Idx...>;
        static_assert(extent_type::rank() == 2);

        tiled_layout_impl::mapping<extent_type> tiled_mapping(extents);
        using view_data_type = std::ranges::range_value_t<decltype(view)>;
        return Kokkos::mdspan<T, extent_type, tiled_layout_impl, Accessor>(std::ranges::data(view), tiled_mapping, acc);
    }

    struct tiled_layout_impl {
        template<class Extents>
        struct mapping {
            // for simplicity
            static_assert(Extents::rank() == 2, "SimpleTileLayout2D is hard-coded for 2D layout");

            using extents_type = Extents;
            using rank_type = typename Extents::rank_type;
            using size_type = typename Extents::size_type;
            using layout_type = tiled_layout;

            mapping() noexcept = default;
            mapping(const mapping &) noexcept = default;
            mapping &operator=(const mapping &) noexcept = default;

            mapping(const Extents &exts, size_type row_tile, size_type col_tile) noexcept
                : m_extents(exts), m_row_tile_size(row_tile), m_col_tile_size(col_tile) {
                // For simplicity, don't worry about negatives/zeros/etc.
                assert(row_tile > 0);
                assert(col_tile > 0);
                assert(exts.extent(0) > 0);
                assert(exts.extent(1) > 0);
            }

            MDSPAN_TEMPLATE_REQUIRES(class OtherExtents,
                /* requires */ (::std::is_constructible<extents_type, OtherExtents>::value))
            MDSPAN_CONDITIONAL_EXPLICIT((!::std::is_convertible<OtherExtents, extents_type>::value))
            constexpr mapping(const mapping<OtherExtents> &input_mapping) noexcept
                : m_extents(input_mapping.extents()), m_row_tile_size(input_mapping.row_tile_size_),
                  m_col_tile_size(input_mapping.col_tile_size_) {}

            //------------------------------------------------------------
            // Helper members (not part of the layout concept)

            constexpr size_type n_row_tiles() const noexcept {
                return m_extents.extent(0) / m_row_tile_size + size_type((m_extents.extent(0) % m_row_tile_size) != 0);
            }

            constexpr size_type n_column_tiles() const noexcept {
                return m_extents.extent(1) / m_col_tile_size + size_type((m_extents.extent(1) % m_col_tile_size) != 0);
            }

            constexpr size_type tile_size() const noexcept { return m_row_tile_size * m_col_tile_size; }

            size_type tile_offset(size_type row, size_type col) const noexcept {
                // This could probably be more efficient, but for example purposes...
                auto col_tile = col / m_col_tile_size;
                auto row_tile = row / m_row_tile_size;
                // We're hard-coding this to *column-major* layout across tiles
                return (col_tile * n_row_tiles() + row_tile) * tile_size();
            }

            size_type offset_in_tile(size_type row, size_type col) const noexcept {
                auto t_row = row % m_row_tile_size;
                auto t_col = col % m_col_tile_size;
                // We're hard-coding this to *row-major* within tiles
                return t_row * m_col_tile_size + t_col;
            }

            //------------------------------------------------------------
            // Required members

            constexpr const extents_type &extents() const { return m_extents; }

            constexpr size_type required_span_size() const noexcept {
                return n_row_tiles() * n_column_tiles() * tile_size();
            }

            template<class RowIndex, class ColIndex>
            // requires(is_convertible_v<RowIndex, size_type> &&
            //   is_convertible_v<ColIndex, size_type> &&
            //   is_nothrow_constructible_v<size_type, RowIndex> &&
            //   is_nothrow_constructible_v<size_type, ColIndex>)
            constexpr size_type operator()(RowIndex row, ColIndex col) const noexcept {
                // TODO (mfh 2022/08/04 check precondition that
                // extents_type::index-cast(row, col)
                // is a multidimensional index in extents_.
                return tile_offset(row, col) + offset_in_tile(row, col);
            }

            // Mapping is always unique
            static constexpr bool is_always_unique() noexcept { return true; }
            // Only exhaustive if extents_.extent(0) % column_tile_size == 0, so not
            // always
            static constexpr bool is_always_exhaustive() noexcept { return false; }
            // There is not always a regular stride between elements in a given
            // dimension
            static constexpr bool is_always_strided() noexcept { return false; }

            static constexpr bool is_unique() noexcept { return true; }
            constexpr bool is_exhaustive() const noexcept {
                // Only exhaustive if extents fit exactly into tile sizes...
                return (m_extents.extent(0) % m_row_tile_size == 0) && (m_extents.extent(1) % m_col_tile_size == 0);
            }
            // There are some circumstances where this is strided, but we're not
            // concerned about that optimization, so we're allowed to just return
            // false here
            constexpr bool is_strided() const noexcept { return false; }

          private:
            Extents m_extents;
            size_type m_row_tile_size = 1;
            size_type m_col_tile_size = 1;
        };
    };
};


struct indexed_layout {
    template<typename Extents>
    static constexpr auto build(
        std::ranges::forward_range auto &&view, Extents extents, std::span<size_t> sizes, std::span<size_t> strides) {
        using T = std::ranges::range_value_t<decltype(view)>;
        return std::move(Kokkos::mdspan<T, Extents, indexed_layout_impl>(
            std::ranges::data(view), indexed_layout_impl::mapping<Extents>(extents, sizes, strides)));
    }

    struct indexed_layout_impl {
        template<typename Extents>
        struct mapping {
            friend struct indexed_layout;

            using extents_type = Extents;
            using rank_type = typename extents_type::rank_type;
            using size_type = typename extents_type::size_type;
            using layout_type = indexed_layout;

            mapping() noexcept = default;
            mapping(const mapping &) noexcept = default;
            mapping &operator=(const mapping &) noexcept = default;

            constexpr mapping(const Extents &ext, std::span<size_t> sizes, std::span<size_t> strides)
                : m_extents(ext), m_sizes(sizes), m_strides(strides) {
                assert(sizes.size() == strides.size() && "Sizes and strides must have the same size");
                m_view_size = std::accumulate(m_sizes.begin(), m_sizes.end(), 0);
                m_partial_sums.resize(m_sizes.size());
                std::partial_sum(m_sizes.begin(), m_sizes.end(), m_partial_sums.begin());
            }

            constexpr const extents_type &extents() const { return m_extents; }

            constexpr size_type required_span_size() const noexcept { return m_view_size; }

            template<typename... Index>
            constexpr size_type operator()(Index... indexes) const noexcept {
                auto linear_idx = details::linear_index(m_extents, indexes...);
                // + 1 to include the first stride
                const auto idx_pos = std::distance(m_partial_sums.begin(),
                                         std::lower_bound(m_partial_sums.begin(), m_partial_sums.end(), linear_idx,
                                             [](const auto &vet_val, const auto &val) { return val >= vet_val; }))
                    + 1;
                std::for_each_n(m_strides.begin(), idx_pos, [&](auto &sum) { linear_idx += sum; });
                return linear_idx;
            }


            static constexpr bool is_always_unique() noexcept { return true; }
            static constexpr bool is_always_exhaustive() noexcept { return false; }
            // There is not always a regular stride between elements in a given
            // dimension
            static constexpr bool is_always_strided() noexcept { return false; }

            static constexpr bool is_unique() noexcept { return true; }
            constexpr bool is_exhaustive() const noexcept {
                return false; // TODO....
            }
            constexpr bool is_strided() const noexcept { return true; }

          private:
            Extents m_extents;
            std::span<size_t> m_sizes;
            std::span<size_t> m_strides;
            size_t m_view_size{};
            std::vector<size_t> m_partial_sums;
        };
    };

    // One-dimensional compact
    template<typename T, template<typename, size_t...> typename Extents, typename Layout, typename Accessor,
        typename IdxType, size_t... Idx>
        requires(Extents<IdxType, Idx...>::rank() == 1)
    static constexpr auto compact(const Kokkos::mdspan<T, Extents<IdxType, Idx...>, Layout, Accessor> &view) {
        using element_type = std::remove_cvref_t<typename Accessor::element_type>;
        auto &mapping = view.mapping();
        auto ptr = new element_type[mapping.required_span_size()];
        auto base = ptr;

        // Iterate over each chunk
        for(size_t i = 0, pos = 0; i < mapping.m_sizes.size(); i++) {
            const auto size = mapping.m_sizes[i];
            // -1 to avoid getting into the other chunk (as std::copy ignores the last element)
            // + 1 to include the include the last element in the chunk
            std::copy(&view(pos), &view(pos + size - 1) + 1, ptr);
            pos += size;
            ptr += size;
        }

        details::conditional_deleter<element_type> del(true);
        return std::unique_ptr<element_type, decltype(del)>(std::move(base), del);
    }
};

struct subarray_layout {
    template<details::order Order = details::row_major, typename Extents>
    static constexpr auto build(std::ranges::forward_range auto &&view, Extents extents,
        const std::array<size_t, Extents::rank()>& strides,
        const std::array<size_t, Extents::rank()>& offsets) {
        using T = std::ranges::range_value_t<decltype(view)>;
        auto linear_index = details::linear_index<Order>(extents, offsets);
        return std::move(Kokkos::mdspan<T, Extents, subarray_layout_impl>(
            std::ranges::data(view) + linear_index, subarray_layout_impl::mapping<Extents>(extents, strides)));
    }

    struct subarray_layout_impl : Kokkos::layout_stride {
        template<typename Extents>
        struct mapping : Kokkos::layout_stride::mapping<Extents> {
            using base = Kokkos::layout_stride::mapping<Extents>;
            using base::base;
        };
    };
};

// ##################### Layouts for benchmarking purposes ##################### //

struct block_layout {
    template<typename Range, template<typename, size_t...> typename Extents, typename T, typename Size, typename Stride,
        size_t... Idx>
        requires std::ranges::forward_range<Range>
    [[nodiscard]] static constexpr auto build(
        Range &&view, Extents<T, Idx...> extents, Size &&blocks, Stride &&strides) {
        // TODO: Check sizes against view
        // TODO: Tiled layout should work on every dimension
        using view_data_type = std::ranges::range_value_t<Range>;
        return build(view, extents, std::forward<Size>(blocks), std::forward<Stride>(strides),
            Kokkos::default_accessor<view_data_type>());
    }

    // Hardwritten for 1D layouts, will change later...
    template<typename Range, template<typename, size_t...> typename Extents, typename T, size_t... Idx,
        typename Accessor>
        requires std::ranges::forward_range<Range>
    [[nodiscard]] static constexpr auto build(Range &&view, Extents<T, Idx...> extents, std::size_t block,
        std::size_t stride, const Accessor &acc = Kokkos::default_accessor<std::ranges::range_value_t<Range>>()) {
        // TODO: Check sizes against view
        // TODO: Tiled layout should work on every dimension
        using extent_type = std::remove_cvref_t<Extents<T, Idx...>>;
        static_assert(extent_type::rank() == 1);

        tiled_block_layout::mapping<extent_type> block_mapping(extents, block, stride);
        using view_data_type = std::ranges::range_value_t<decltype(view)>;
        return Kokkos::mdspan<T, extent_type, typename decltype(block_mapping)::layout_type, Accessor>(
            std::ranges::data(view), block_mapping, acc);
    }

    // Hardwritten for 1D layouts, will change later...
    template<typename Range, template<typename, size_t...> typename Extents, typename T, size_t... Idx,
        typename Accessor>
        requires std::ranges::forward_range<Range>
    [[nodiscard]] static constexpr auto build(Range &&view, Extents<T, Idx...> extents,
        const std::span<std::size_t, 2> &blocks, std::size_t stride,
        const Accessor &acc = Kokkos::default_accessor<std::ranges::range_value_t<Range>>()) {
        // TODO: Check sizes against view
        // TODO: Tiled layout should work on every dimension
        using extent_type = std::remove_cvref_t<Extents<T, Idx...>>;
        static_assert(extent_type::rank() == 1);

        bucket_block_layout::mapping<extent_type> block_mapping(extents, blocks, stride);
        using view_data_type = std::ranges::range_value_t<decltype(view)>;
        return Kokkos::mdspan<T, extent_type, typename decltype(block_mapping)::layout_type, Accessor>(
            std::ranges::data(view), block_mapping, acc);
    }

    // Hardwritten for 1D layouts, will change later...
    template<typename Range, template<typename, size_t...> typename Extents, typename T, size_t... Idx,
        typename Accessor>
        requires std::ranges::forward_range<Range>
    [[nodiscard]] static constexpr auto build(Range &&view, Extents<T, Idx...> extents, std::size_t block,
        const std::span<std::size_t, 2> &strides,
        const Accessor &acc = Kokkos::default_accessor<std::ranges::range_value_t<Range>>()) {
        // TODO: Check sizes against view
        // TODO: Tiled layout should work on every dimension
        using extent_type = std::remove_cvref_t<Extents<T, Idx...>>;
        static_assert(extent_type::rank() == 1);

        block_block_layout::mapping<extent_type> block_mapping(extents, block, strides);
        using view_data_type = std::ranges::range_value_t<decltype(view)>;
        return Kokkos::mdspan<T, extent_type, typename decltype(block_mapping)::layout_type, Accessor>(
            std::ranges::data(view), block_mapping, acc);
    }

    // Hardwritten for 1D layouts, will change later...
    template<typename Range, template<typename, size_t...> typename Extents, typename T, size_t... Idx,
        typename Accessor>
        requires std::ranges::forward_range<Range>
    [[nodiscard]] static constexpr auto build(Range &&view, Extents<T, Idx...> extents,
        const std::span<std::size_t, 2> &block, const std::span<std::size_t, 2> &stride,
        const Accessor &acc = Kokkos::default_accessor<std::ranges::range_value_t<Range>>()) {
        // TODO: Check sizes against view
        // TODO: Tiled layout should work on every dimension
        using extent_type = std::remove_cvref_t<Extents<T, Idx...>>;
        static_assert(extent_type::rank() == 1);

        alternating_block_layout::mapping<extent_type> block_mapping(extents, block, stride);
        using view_data_type = std::ranges::range_value_t<decltype(view)>;
        return Kokkos::mdspan<T, extent_type, typename decltype(block_mapping)::layout_type, Accessor>(
            std::ranges::data(view), block_mapping, acc);
    }

    struct tiled_block_layout {
        template<typename Extents>
        struct mapping {
            friend class block_layout;
            friend class tiled_block_layout;

            static_assert(Extents::rank() == 1, "Hardwritten for 1D layouts, for now");

            using extents_type = Extents;
            using rank_type = typename extents_type::rank_type;
            using size_type = typename extents_type::size_type;
            using layout_type = tiled_block_layout;

            mapping() noexcept = default;
            mapping(const mapping &) noexcept = default;
            mapping &operator=(const mapping &) noexcept = default;

            constexpr mapping(const Extents &ext, std::size_t size, std::size_t stride)
                : m_extents(ext), m_size(size), m_stride(stride) {
                assert(m_size <= stride);
                m_offset = stride - m_size;
            }

            // TODO: copy constructor

            // Mandatory member methods

            constexpr const extents_type &extents() const { return m_extents; }

            constexpr size_type required_span_size() const noexcept {
                // Kokkos::extents<int, 1> x;
                return m_extents.extent(0);
            }

            template<class Index>
            constexpr size_type operator()(Index idx) const noexcept {
                return idx + (idx / m_size * m_offset);
            }

            // Mapping is always unique
            static constexpr bool is_always_unique() noexcept { return true; }
            // Only exhaustive if extents_.extent(0) % column_tile_size == 0, so not
            // always
            static constexpr bool is_always_exhaustive() noexcept { return false; }
            // There is not always a regular stride between elements in a given
            // dimension
            static constexpr bool is_always_strided() noexcept { return false; }

            static constexpr bool is_unique() noexcept { return true; }
            constexpr bool is_exhaustive() const noexcept {
                // Only exhaustive if extents fit exactly into tile sizes...
                // return (extents_.extent(0) % row_tile_size_ == 0) &&
                //  (extents_.extent(1) % col_tile_size_ == 0);
                return false; // TODO....
            }
            // There are some circumstances where this is strided, but we're not
            // concerned about that optimization, so we're allowed to just return
            // false here
            constexpr bool is_strided() const noexcept { return true; }

          private:
            Extents m_extents;
            std::size_t m_size{};
            std::size_t m_stride{};
            std::size_t m_offset{};
        };
    };

    struct block_block_layout {
        template<typename Extents>
        struct mapping {
            friend class block_layout;
            friend class block_block_layout;

            static_assert(Extents::rank() == 1, "Hardwritten for 1D layouts, for now");

            using extents_type = Extents;
            using rank_type = typename extents_type::rank_type;
            using size_type = typename extents_type::size_type;
            using layout_type = block_block_layout;

            mapping() noexcept = default;
            mapping(const mapping &) noexcept = default;
            mapping &operator=(const mapping &) noexcept = default;

            constexpr mapping(const Extents &ext, const std::size_t size, const std::span<std::size_t, 2> &strides)
                : m_extents(ext), m_strides(strides), m_size(size) {
                assert(m_strides[0] >= m_size);
                assert(m_strides[1] >= m_size);
                m_offsets[0] = strides[0] - size;
                m_offsets[1] = strides[1] - size;
            }

            // Mandatory member methods

            constexpr const extents_type &extents() const { return m_extents; }

            constexpr size_type required_span_size() const noexcept {
                // Kokkos::extents<int, 1> x;
                return m_extents.extent(0);
            }

            template<class Index>
            constexpr size_type operator()(Index idx) const noexcept {
                auto block = idx / m_size;
                auto num_blocks = block / 2;
                auto remainder = block % 2;
                return idx + (num_blocks + remainder) * m_offsets[0] + num_blocks * m_offsets[1];
            }

            // Mapping is always unique
            static constexpr bool is_always_unique() noexcept { return true; }
            // Only exhaustive if extents_.extent(0) % column_tile_size == 0, so not
            // always
            static constexpr bool is_always_exhaustive() noexcept { return false; }
            // There is not always a regular stride between elements in a given
            // dimension
            static constexpr bool is_always_strided() noexcept { return false; }

            static constexpr bool is_unique() noexcept { return true; }
            constexpr bool is_exhaustive() const noexcept {
                // Only exhaustive if extents fit exactly into tile sizes...
                // return (extents_.extent(0) % row_tile_size_ == 0) &&
                //  (extents_.extent(1) % col_tile_size_ == 0);
                return false; // TODO....
            }
            // There are some circumstances where this is strided, but we're not
            // concerned about that optimization, so we're allowed to just return
            // false here
            constexpr bool is_strided() const noexcept { return true; }

          private:
            Extents m_extents;
            std::size_t m_size;
            std::span<std::size_t, 2> m_strides;
            std::array<std::size_t, 2> m_offsets;
        };
    };

    struct bucket_block_layout {
        template<typename Extents>
        struct mapping {
            friend class block_layout;
            friend class bucket_block_layout;

            static_assert(Extents::rank() == 1, "Hardwritten for 1D layouts, for now");

            using extents_type = Extents;
            using rank_type = typename extents_type::rank_type;
            using size_type = typename extents_type::size_type;
            using layout_type = bucket_block_layout;

            mapping() noexcept = default;
            mapping(const mapping &) noexcept = default;
            mapping &operator=(const mapping &) noexcept = default;

            constexpr mapping(const Extents &ext, const std::span<size_t, 2> &sizes, size_t stride)
                : m_extents(ext), m_sizes(sizes), m_stride(stride) {
                assert(m_stride >= m_sizes[0]);
                assert(m_stride >= m_sizes[1]);
                // At least one block and stride
                m_diffs[0] = m_stride - m_sizes[0];
                m_diffs[1] = m_stride - m_sizes[1];
                m_sum_blocks += m_sizes[0] + m_sizes[1];
                m_sum_diffs += m_diffs[0] + m_diffs[1];

                m_offset = m_diffs[0];
                // std::inclusive_scan(diffs.begin(), diffs.end() - 1, offsets.begin());
                m_partial_blocks[0] = m_sizes[0];
                m_partial_blocks[1] = m_sizes[0] + m_sizes[1];
            }

            // Mandatory member methods

            constexpr const extents_type &extents() const { return m_extents; }

            constexpr size_type required_span_size() const noexcept {
                // Kokkos::extents<int, 1> x;
                return m_extents.extent(0);
            }

            template<class Index>
            constexpr inline size_t pos_in_blocks(Index idx) const noexcept {
                return idx / m_sum_blocks;
            }

            template<class Index>
            constexpr inline size_t offset_in_block(Index idx) const noexcept {
                const auto pos = idx % m_sum_blocks;
                return pos < m_sizes[0] ? 0 : m_offset;
            }

            template<class Index>
            constexpr size_type operator()(Index idx) const noexcept {
                return idx + m_sum_diffs * pos_in_blocks(idx) + offset_in_block(idx);
            }

            // Mapping is always unique
            static constexpr bool is_always_unique() noexcept { return true; }
            // Only exhaustive if extents_.extent(0) % column_tile_size == 0, so not
            // always
            static constexpr bool is_always_exhaustive() noexcept { return false; }
            // There is not always a regular stride between elements in a given
            // dimension
            static constexpr bool is_always_strided() noexcept { return false; }

            static constexpr bool is_unique() noexcept { return true; }
            constexpr bool is_exhaustive() const noexcept {
                // Only exhaustive if extents fit exactly into tile sizes...
                // return (extents_.extent(0) % row_tile_size_ == 0) &&
                //  (extents_.extent(1) % col_tile_size_ == 0);
                return false; // TODO....
            }
            // There are some circumstances where this is strided, but we're not
            // concerned about that optimization, so we're allowed to just return
            // false here
            constexpr bool is_strided() const noexcept { return true; }

          private:
            Extents m_extents;
            std::span<size_t> m_sizes;
            size_t m_stride;
            std::array<size_t, 2> m_partial_blocks;
            std::array<size_t, 2> m_diffs;
            size_t m_offset;
            size_t m_sum_diffs{};
            size_t m_sum_blocks{};
        };
    };

    struct alternating_block_layout {
        template<typename Extents>
        struct mapping {
            friend class block_layout;
            friend class alternating_block_layout;

            static_assert(Extents::rank() == 1, "Hardwritten for 1D layouts, for now");

            using extents_type = Extents;
            using rank_type = typename extents_type::rank_type;
            using size_type = typename extents_type::size_type;
            using layout_type = alternating_block_layout;

            mapping() noexcept = default;
            mapping(const mapping &) noexcept = default;
            mapping &operator=(const mapping &) noexcept = default;

            constexpr mapping(
                const Extents &ext, const std::span<size_t, 2> &sizes, const std::span<size_t, 2> &strides)
                : m_extents(ext), m_sizes(sizes), m_strides(strides), m_sum_blocks(0), m_sum_diffs(0) {
                // At least one block and stride
                m_diffs[0] = m_strides[0] - m_sizes[0];
                m_diffs[1] = m_strides[1] - m_sizes[1];
                m_sum_blocks += m_sizes[0] + m_sizes[1];
                m_sum_diffs += m_diffs[0] + m_diffs[1];

                m_offset = m_diffs[0];
                m_partial_blocks[0] = m_sizes[0];
                m_partial_blocks[1] = m_sizes[0] + m_sizes[1];
            }

            // TODO: copy constructor

            // Mandatory member methods

            constexpr const extents_type &extents() const { return m_extents; }

            constexpr size_type required_span_size() const noexcept { return m_extents.extent(0); }

            template<class Index>
            constexpr inline size_t pos_in_blocks(Index idx) const noexcept {
                return idx / m_sum_blocks;
            }

            template<class Index>
            constexpr inline size_t offset_in_block(Index idx) const noexcept {
                const auto pos = idx % m_sum_blocks;
                return pos < m_sizes[0] ? 0 : m_offset;
            }

            template<class Index>
            constexpr size_type operator()(Index idx) const noexcept {
                return idx + m_sum_diffs * pos_in_blocks(idx) + offset_in_block(idx);
            }

            // Mapping is always unique
            static constexpr bool is_always_unique() noexcept { return true; }
            // Only exhaustive if extents_.extent(0) % column_tile_size == 0, so not
            // always
            static constexpr bool is_always_exhaustive() noexcept { return false; }
            // There is not always a regular stride between elements in a given
            // dimension
            static constexpr bool is_always_strided() noexcept { return false; }

            static constexpr bool is_unique() noexcept { return true; }
            constexpr bool is_exhaustive() const noexcept {
                // Only exhaustive if extents fit exactly into tile sizes...
                // return (extents_.extent(0) % row_tile_size_ == 0) &&
                //  (extents_.extent(1) % col_tile_size_ == 0);
                return false; // TODO....
            }
            // There are some circumstances where this is strided, but we're not
            // concerned about that optimization, so we're allowed to just return
            // false here
            constexpr bool is_strided() const noexcept { return true; }

          private:
            Extents m_extents;
            std::span<size_t> m_sizes;
            std::span<size_t> m_strides;
            std::array<size_t, 2> m_partial_blocks;
            std::array<size_t, 2> m_diffs;
            size_t m_offset;
            size_t m_sum_diffs{};
            size_t m_sum_blocks{};
        };
    };

    // One-dimensional compact
    template<typename T, template<typename, size_t...> typename Extents, typename Layout, typename Accessor,
        typename IdxType, size_t... Idx>
        requires(sizeof...(Idx) == 1)
    static constexpr auto compact(const Kokkos::mdspan<T, Extents<IdxType, Idx...>, Layout, Accessor> &view) {
        using element_type = std::remove_cvref_t<typename Accessor::element_type>;

        auto ptr = new element_type[view.size()];
        auto base = ptr;
        using extent_type = typename std::remove_cvref_t<decltype(view)>::extents_type;
        auto &mapping = view.mapping();

        auto compact_f = [&](std::ranges::forward_range auto &&sizes) {
            int num_blocks = sizes.size();
            for(int pos = 0, block = 0; pos < view.extent(0); pos += sizes[block], block = (block + 1) % num_blocks) {
                std::copy(&view[pos], &view[pos] + sizes[block], ptr);
                ptr += sizes[block];
            }
            details::conditional_deleter<element_type> del(true);
            return std::unique_ptr<element_type, decltype(del)>(std::move(base), del);
        };

        if constexpr(std::is_same_v<Layout, bucket_block_layout> || std::is_same_v<Layout, alternating_block_layout>) {
            return compact_f(mapping.m_sizes);
        } else {
            return compact_f(std::span<const std::size_t>(&mapping.m_size, 1));
        }
    }
};

} // namespace empi::layouts

namespace empi::details {

template<typename K>
concept is_block_layout = std::is_same_v<K, layouts::block_layout::tiled_block_layout>
    || std::is_same_v<K, layouts::block_layout::bucket_block_layout>
    || std::is_same_v<K, layouts::block_layout::block_block_layout>
    || std::is_same_v<K, layouts::block_layout::alternating_block_layout>;
}

#endif

// struct block_layout {

//   template <template <typename, size_t...> typename Extents, typename T,
//             size_t... idx>
//   [[nodiscard]] static constexpr auto
//   build(std::ranges::forward_range auto &&view, Extents<T, idx...> extents,
//         std::ranges::forward_range auto &&blocks,
//         std::ranges::forward_range auto &&strides) {
//     // TODO: Check sizes against view
//     // TODO: Tiled layout should work on every dimension
//     using view_data_type = std::ranges::range_value_t<decltype(view)>;
//     return build(view, extents, blocks, strides,
//                 Kokkos::default_accessor<view_data_type>());
//   }

//   // Hardwritten for 1D layouts, will change later...
//   template <template <typename, size_t...> typename Extents, typename T,
//             size_t... idx, typename Accessor>
//   [[nodiscard]] static constexpr auto
//   build(std::ranges::forward_range auto &&view, Extents<T, idx...> extents,
//         std::ranges::forward_range auto &&blocks,
//         std::ranges::forward_range auto &&strides,
//         const Accessor &acc = Kokkos::default_accessor<
//             std::ranges::range_value_t<decltype(view)>>()) {
//     // TODO: Check sizes against view
//     // TODO: Tiled layout should work on every dimension
//     using extent_type = std::remove_cvref_t<Extents<T, idx...>>;
//     static_assert(extent_type::rank() == 1);

//     mapping<extent_type> block_mapping(extents, blocks, strides);
//     using view_data_type = std::ranges::range_value_t<decltype(view)>;
//     return Kokkos::mdspan<T, extent_type, block_layout, Accessor>(
//         std::ranges::data(view), block_mapping, acc);
//   }

//   /**
//         If blocked layout, segmented copy
// */
//   template <typename T, template <typename, size_t...> typename Extents,
//             typename Accessor, typename idx_type, size_t... idx>
//   static constexpr auto
//   compact(const Kokkos::mdspan<T, Extents<idx_type, idx...>, block_layout,
//                               Accessor> &view) {
//     using element_type = std::remove_cvref_t<typename
//     Accessor::element_type>;

//     auto ptr = new element_type[view.size()];
//     auto base = ptr;
//     using extent_type =
//         typename std::remove_cvref_t<decltype(view)>::extents_type;
//     const auto &mapping = view.mapping();
//     int num_blocks = mapping.blocks.size();

//     for (int pos = 0, block = 0; pos < view.extent(0);
//          pos += mapping.blocks[block], block = (block + 1) % num_blocks) {
//       std::copy(&view[pos], &view[pos] + mapping.blocks[block], ptr);
//       ptr += mapping.blocks[block];
//     }
//     details::conditional_deleter<element_type> del(true);
//     return std::unique_ptr<element_type, decltype(del)>(std::move(base),
//     del);
//   }

//   template <typename Extents> struct mapping {
//     friend class block_layout;

//     static_assert(Extents::rank() == 1, "Hardwritten for 1D layouts, for
//     now");

//     using extents_type = Extents;
//     using rank_type = typename extents_type::rank_type;
//     using size_type = typename extents_type::size_type;
//     using layout_type = block_layout;

//     mapping() noexcept = default;
//     mapping(const mapping &) noexcept = default;
//     mapping &operator=(const mapping &) noexcept = default;

//     constexpr mapping(const Extents &ext, std::ranges::forward_range auto
//     &&b,
//                       std::ranges::forward_range auto &&s)
//         : _extents(ext), blocks(b), strides(s), sum_blocks(0), sum_diffs(0) {
//       // At least one block and stride
//       for (int i = 0; i < blocks.size(); i++) {
//         diffs.push_back(strides[i] - blocks[i]);
//         sum_blocks += blocks[i];
//         sum_diffs += diffs[i];
//       }

//       offsets.resize(diffs.size() - 1);
//       std::inclusive_scan(diffs.begin(), diffs.end() - 1, offsets.begin());

//       partial_blocks.resize(blocks.size());
//       std::inclusive_scan(blocks.begin(), blocks.end(),
//       partial_blocks.begin());
//     }

//     // TODO: copy constructor

//     // Mandatory member methods

//     constexpr const extents_type &extents() const { return _extents; }

//     constexpr size_type required_span_size() const noexcept {
//       // Kokkos::extents<int, 1> x;
//       return _extents.extent(0);
//     }

//     template <class index>
//     constexpr inline size_t pos_in_blocks(index idx) const noexcept {
//       return idx / sum_blocks;
//     }

//     template <class index>
//     constexpr inline size_t offset_in_block(index idx) const noexcept {
//       const auto pos = idx % sum_blocks;
//       if (pos < blocks[0])
//         return 0;

//       const auto new_pos =
//           std::distance(partial_blocks.begin(),
//                         std::find_if(partial_blocks.begin(),
//                                      partial_blocks.end(),
//                                      [pos](auto &e) { return pos < e; })) -
//           1;
//       return offsets[new_pos];
//     }

//     template <class index>
//     constexpr size_type operator()(index idx) const noexcept {
//       return idx + sum_diffs * pos_in_blocks(idx) + offset_in_block(idx);
//     }

//     // Mapping is always unique
//     static constexpr bool is_always_unique() noexcept { return true; }
//     // Only exhaustive if extents_.extent(0) % column_tile_size == 0, so not
//     // always
//     static constexpr bool is_always_exhaustive() noexcept { return false; }
//     // There is not always a regular stride between elements in a given
//     // dimension
//     static constexpr bool is_always_strided() noexcept { return false; }

//     static constexpr bool is_unique() noexcept { return true; }
//     constexpr bool is_exhaustive() const noexcept {
//       // Only exhaustive if extents fit exactly into tile sizes...
//       // return (extents_.extent(0) % row_tile_size_ == 0) &&
//       //  (extents_.extent(1) % col_tile_size_ == 0);
//       return false; // TODO....
//     }
//     // There are some circumstances where this is strided, but we're not
//     // concerned about that optimization, so we're allowed to just return
//     // false here
//     constexpr bool is_strided() const noexcept { return true; }

//   private:
//     Extents _extents;
//     std::span<size_t> blocks;
//     std::span<size_t> strides;
//     std::vector<size_t> partial_blocks;
//     std::vector<size_t> diffs;
//     std::vector<size_t> offsets;
//     size_t sum_diffs{};
//     size_t sum_blocks{};
//   };
// };

// } // namespace empi::layouts

// #endif /* INCLUDE_EMPI_LAYOUTS */
