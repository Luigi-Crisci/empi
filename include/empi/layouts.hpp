#ifndef INCLUDE_EMPI_LAYOUTS
#define INCLUDE_EMPI_LAYOUTS

#include <cassert>
#include <memory>
#include <utility>

#include <empi/datatype.hpp>
#include <empi/defines.hpp>


namespace empi::layouts {

struct column_layout {
    // Hardwritten for 2D layouts, will change later...
    template<template<typename, size_t...> typename Extents, typename K, size_t... Idx>
    [[nodiscard]] static auto build(std::ranges::forward_range auto &&view, Extents<K, Idx...> extents, size_t col) {
        return column_layout::build(
            view, extents, col, Kokkos::default_accessor<std::ranges::range_value_t<decltype(view)>>());
    }

    // Hardwritten for 2D layouts, will change later...
    template<template<typename, size_t...> typename Extents, typename K, size_t... Idx, typename Accessor>
    [[nodiscard]] static auto build(
        std::ranges::forward_range auto &&view, Extents<K, Idx...> extents, size_t col, const Accessor &acc) {
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

struct tiled_layout {
    // Hardwritten for 2D layouts, will change later...
    [[nodiscard]] static constexpr auto build(
        std::ranges::forward_range auto &&view, const size_t row, const size_t col) {
        return tiled_layout::build(view, Kokkos::dextents<size_t, 2>(row, col),
            Kokkos::default_accessor<std::ranges::range_value_t<decltype(view)>>());
    }

    // Hardwritten for 2D layouts, will change later...
    template<template<typename, size_t...> typename Extents, typename T, size_t... idx, typename Accessor>
    [[nodiscard]] static constexpr auto build(
        std::ranges::forward_range auto &&view, Extents<T, idx...> extents, const Accessor &acc) {
        // TODO: Check sizes against view
        // TODO: Tiled layout should work on every dimension
        using extent_type = Extents<T, idx...>;
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
                : extents_(exts), row_tile_size_(row_tile), col_tile_size_(col_tile) {
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
                : extents_(input_mapping.extents()), row_tile_size_(input_mapping.row_tile_size_),
                  col_tile_size_(input_mapping.col_tile_size_) {}

            //------------------------------------------------------------
            // Helper members (not part of the layout concept)

            constexpr size_type n_row_tiles() const noexcept {
                return extents_.extent(0) / row_tile_size_ + size_type((extents_.extent(0) % row_tile_size_) != 0);
            }

            constexpr size_type n_column_tiles() const noexcept {
                return extents_.extent(1) / col_tile_size_ + size_type((extents_.extent(1) % col_tile_size_) != 0);
            }

            constexpr size_type tile_size() const noexcept { return row_tile_size_ * col_tile_size_; }

            size_type tile_offset(size_type row, size_type col) const noexcept {
                // This could probably be more efficient, but for example purposes...
                auto col_tile = col / col_tile_size_;
                auto row_tile = row / row_tile_size_;
                // We're hard-coding this to *column-major* layout across tiles
                return (col_tile * n_row_tiles() + row_tile) * tile_size();
            }

            size_type offset_in_tile(size_type row, size_type col) const noexcept {
                auto t_row = row % row_tile_size_;
                auto t_col = col % col_tile_size_;
                // We're hard-coding this to *row-major* within tiles
                return t_row * col_tile_size_ + t_col;
            }

            //------------------------------------------------------------
            // Required members

            constexpr const extents_type &extents() const { return extents_; }

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
                return (extents_.extent(0) % row_tile_size_ == 0) && (extents_.extent(1) % col_tile_size_ == 0);
            }
            // There are some circumstances where this is strided, but we're not
            // concerned about that optimization, so we're allowed to just return
            // false here
            constexpr bool is_strided() const noexcept { return false; }

          private:
            Extents extents_;
            size_type row_tile_size_ = 1;
            size_type col_tile_size_ = 1;
        };
    };
};

//////////// Layouts for benchmarking purposes ///////////////

struct block_layout {
    template<template<typename, size_t...> typename Extents, typename T, typename Size, typename Stride, size_t... idx>
    [[nodiscard]] static constexpr auto build(
        std::ranges::forward_range auto &&view, Extents<T, idx...> extents, Size &&blocks, Stride &&strides) {
        // TODO: Check sizes against view
        // TODO: Tiled layout should work on every dimension
        using view_data_type = std::ranges::range_value_t<decltype(view)>;
        return build(view, extents, std::forward<Size>(blocks), std::forward<Stride>(strides),
            Kokkos::default_accessor<view_data_type>());
    }

    // Hardwritten for 1D layouts, will change later...
    template<template<typename, size_t...> typename Extents, typename T, size_t... idx, typename Accessor>
    [[nodiscard]] static constexpr auto build(std::ranges::forward_range auto &&view, Extents<T, idx...> extents,
        std::size_t block, std::size_t stride,
        const Accessor &acc = Kokkos::default_accessor<std::ranges::range_value_t<decltype(view)>>()) {
        // TODO: Check sizes against view
        // TODO: Tiled layout should work on every dimension
        using extent_type = std::remove_cvref_t<Extents<T, idx...>>;
        static_assert(extent_type::rank() == 1);

        tiled_block_layout::mapping<extent_type> block_mapping(extents, block, stride);
        using view_data_type = std::ranges::range_value_t<decltype(view)>;
        return Kokkos::mdspan<T, extent_type, typename decltype(block_mapping)::layout_type, Accessor>(
            std::ranges::data(view), block_mapping, acc);
    }

    // Hardwritten for 1D layouts, will change later...
    template<template<typename, size_t...> typename Extents, typename T, size_t... idx, typename Accessor>
    [[nodiscard]] static constexpr auto build(std::ranges::forward_range auto &&view, Extents<T, idx...> extents,
        const std::span<std::size_t, 2> &blocks, std::size_t stride,
        const Accessor &acc = Kokkos::default_accessor<std::ranges::range_value_t<decltype(view)>>()) {
        // TODO: Check sizes against view
        // TODO: Tiled layout should work on every dimension
        using extent_type = std::remove_cvref_t<Extents<T, idx...>>;
        static_assert(extent_type::rank() == 1);

        bucket_block_layout::mapping<extent_type> block_mapping(extents, blocks, stride);
        using view_data_type = std::ranges::range_value_t<decltype(view)>;
        return Kokkos::mdspan<T, extent_type, typename decltype(block_mapping)::layout_type, Accessor>(
            std::ranges::data(view), block_mapping, acc);
    }

    // Hardwritten for 1D layouts, will change later...
    template<template<typename, size_t...> typename Extents, typename T, size_t... idx, typename Accessor>
    [[nodiscard]] static constexpr auto build(std::ranges::forward_range auto &&view, Extents<T, idx...> extents,
        std::size_t block, const std::span<std::size_t, 2> &strides,
        const Accessor &acc = Kokkos::default_accessor<std::ranges::range_value_t<decltype(view)>>()) {
        // TODO: Check sizes against view
        // TODO: Tiled layout should work on every dimension
        using extent_type = std::remove_cvref_t<Extents<T, idx...>>;
        static_assert(extent_type::rank() == 1);

        block_block_layout::mapping<extent_type> block_mapping(extents, block, strides);
        using view_data_type = std::ranges::range_value_t<decltype(view)>;
        return Kokkos::mdspan<T, extent_type, typename decltype(block_mapping)::layout_type, Accessor>(
            std::ranges::data(view), block_mapping, acc);
    }

    // Hardwritten for 1D layouts, will change later...
    template<template<typename, size_t...> typename Extents, typename T, size_t... idx, typename Accessor>
    [[nodiscard]] static constexpr auto build(std::ranges::forward_range auto &&view, Extents<T, idx...> extents,
        const std::span<std::size_t, 2> &block, const std::span<std::size_t, 2> &stride,
        const Accessor &acc = Kokkos::default_accessor<std::ranges::range_value_t<decltype(view)>>()) {
        // TODO: Check sizes against view
        // TODO: Tiled layout should work on every dimension
        using extent_type = std::remove_cvref_t<Extents<T, idx...>>;
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
                : _extents(ext), _size(size), _stride(stride) {
                assert(_size <= stride);
                _offset = stride - _size;
            }

            // TODO: copy constructor

            // Mandatory member methods

            constexpr const extents_type &extents() const { return _extents; }

            constexpr size_type required_span_size() const noexcept {
                // Kokkos::extents<int, 1> x;
                return _extents.extent(0);
            }

            template<class index>
            constexpr size_type operator()(index idx) const noexcept {
                return idx + (idx / _size * _offset);
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
            Extents _extents;
            std::size_t _size{};
            std::size_t _stride{};
            std::size_t _offset{};
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
                : _extents(ext), _strides(strides), _size(size) {
                assert(_strides[0] >= _size);
                assert(_strides[1] >= _size);
                _offsets[0] = strides[0] - size;
                _offsets[1] = strides[1] - size;
            }

            // Mandatory member methods

            constexpr const extents_type &extents() const { return _extents; }

            constexpr size_type required_span_size() const noexcept {
                // Kokkos::extents<int, 1> x;
                return _extents.extent(0);
            }

            template<class index>
            constexpr size_type operator()(index idx) const noexcept {
                auto block = idx / _size;
                auto num_blocks = block / 2;
                auto remainder = block % 2;
                return idx + (num_blocks + remainder) * _offsets[0] + num_blocks * _offsets[1];
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
            Extents _extents;
            std::size_t _size;
            std::span<std::size_t, 2> _strides;
            std::array<std::size_t, 2> _offsets;
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
                : _extents(ext), _sizes(sizes), _stride(stride) {
                assert(_stride >= _sizes[0]);
                assert(_stride >= _sizes[1]);
                // At least one block and stride
                _diffs[0] = _stride - _sizes[0];
                _diffs[1] = _stride - _sizes[1];
                _sum_blocks += _sizes[0] + _sizes[1];
                _sum_diffs += _diffs[0] + _diffs[1];

                _offset = _diffs[0];
                // std::inclusive_scan(diffs.begin(), diffs.end() - 1, offsets.begin());
                _partial_blocks[0] = _sizes[0];
                _partial_blocks[1] = _sizes[0] + _sizes[1];
            }

            // Mandatory member methods

            constexpr const extents_type &extents() const { return _extents; }

            constexpr size_type required_span_size() const noexcept {
                // Kokkos::extents<int, 1> x;
                return _extents.extent(0);
            }

            template<class index>
            constexpr inline size_t pos_in_blocks(index idx) const noexcept {
                return idx / _sum_blocks;
            }

            template<class index>
            constexpr inline size_t offset_in_block(index idx) const noexcept {
                const auto pos = idx % _sum_blocks;
                return pos < _sizes[0] ? 0 : _offset;
            }

            template<class index>
            constexpr size_type operator()(index idx) const noexcept {
                return idx + _sum_diffs * pos_in_blocks(idx) + offset_in_block(idx);
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
            Extents _extents;
            std::span<size_t> _sizes;
            size_t _stride;
            std::array<size_t, 2> _partial_blocks;
            std::array<size_t, 2> _diffs;
            size_t _offset;
            size_t _sum_diffs{};
            size_t _sum_blocks{};
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
                : _extents(ext), _sizes(sizes), _strides(strides), _sum_blocks(0), _sum_diffs(0) {
                // At least one block and stride
                _diffs[0] = _strides[0] - _sizes[0];
                _diffs[1] = _strides[1] - _sizes[1];
                _sum_blocks += _sizes[0] + _sizes[1];
                _sum_diffs += _diffs[0] + _diffs[1];

                _offset = _diffs[0];
                _partial_blocks[0] = _sizes[0];
                _partial_blocks[1] = _sizes[0] + _sizes[1];
            }

            // TODO: copy constructor

            // Mandatory member methods

            constexpr const extents_type &extents() const { return _extents; }

            constexpr size_type required_span_size() const noexcept { return _extents.extent(0); }

            template<class index>
            constexpr inline size_t pos_in_blocks(index idx) const noexcept {
                return idx / _sum_blocks;
            }

            template<class index>
            constexpr inline size_t offset_in_block(index idx) const noexcept {
                const auto pos = idx % _sum_blocks;
                return pos < _sizes[0] ? 0 : _offset;
            }

            template<class index>
            constexpr size_type operator()(index idx) const noexcept {
                return idx + _sum_diffs * pos_in_blocks(idx) + offset_in_block(idx);
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
            Extents _extents;
            std::span<size_t> _sizes;
            std::span<size_t> _strides;
            std::array<size_t, 2> _partial_blocks;
            std::array<size_t, 2> _diffs;
            size_t _offset;
            size_t _sum_diffs{};
            size_t _sum_blocks{};
        };
    };

    template<typename T, template<typename, size_t...> typename Extents, typename Layout, typename Accessor,
        typename idx_type, size_t... idx>
    static constexpr auto compact(const Kokkos::mdspan<T, Extents<idx_type, idx...>, Layout, Accessor> &view) {
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
            return compact_f(mapping._sizes);
        } else {
            return compact_f(std::span<const std::size_t>(&mapping._size, 1));
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
