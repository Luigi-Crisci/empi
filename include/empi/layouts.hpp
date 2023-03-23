#ifndef INCLUDE_EMPI_LAYOUTS
#define INCLUDE_EMPI_LAYOUTS

#include <cassert>
#include <functional>
#include <memory>
#include <utility>

#include <empi/datatype.hpp>
#include <empi/defines.hpp>

namespace empi::layouts {

struct column_layout {

  // Hardwritten for 2D layouts, will change later...
  template <template <typename, size_t...> typename Extents, typename K,
            size_t... Idx>
  [[nodiscard]] static auto build(std::ranges::forward_range auto &&view,
                                  Extents<K, Idx...> extents, size_t col) {
    return column_layout::build(
        view, extents, col,
        stdex::default_accessor<std::ranges::range_value_t<decltype(view)>>());
  }

  // Hardwritten for 2D layouts, will change later...
  template <template <typename, size_t...> typename Extents, typename K,
            size_t... Idx, typename Accessor>
  [[nodiscard]] static auto build(std::ranges::forward_range auto &&view,
                                  Extents<K, Idx...> extents, size_t col,
                                  const Accessor &acc) {
    static_assert(Extents<K, Idx...>::rank() == 2);
    assert(col < extents.extent(1));
    using extent_type = decltype(details::remove_last(extents));
    extent_type new_extents(extents.extent(0));
    std::array stride{extents.extent(1)};

    column_layout_impl::mapping<extent_type> column_map(new_extents, stride);

    using T = std::ranges::range_value_t<decltype(view)>;

    return stdex::mdspan<T, extent_type, column_layout_impl, Accessor>(
        std::ranges::data(view) + col, column_map, acc);
  }

  struct column_layout_impl {

    template <typename Extents>
    struct mapping : stdex::layout_stride::mapping<Extents> {
      using base = stdex::layout_stride::mapping<Extents>;
      using base::base;
    };
  };
};

struct contiguous_layout {
  // Wraps data into plain, 1D mdspan.
  // Used for implicit conversions
  [[nodiscard]] static auto build(std::ranges::forward_range auto &&view) {
    using extent_type = stdex::dextents<std::size_t, 1>;
    extent_type extents(std::ranges::size(view));
    using T = std::ranges::range_value_t<decltype(view)>;
    return std::move(stdex::mdspan<T, extent_type, contiguous_layout_impl>(
        std::ranges::data(view), extents));
  }

  template <typename Accessor>
  [[nodiscard]] static auto build(std::ranges::forward_range auto &&view,
                                  const Accessor &acc) {
    using extent_type = stdex::dextents<std::size_t, 1>;
    extent_type extents(std::ranges::size(view));
    using T = std::ranges::range_value_t<decltype(view)>;
    return std::move(
        stdex::mdspan<T, extent_type, contiguous_layout_impl, Accessor>(
            std::ranges::data(view),
            contiguous_layout_impl::mapping<extent_type>(extents), acc));
  }

  struct contiguous_layout_impl {
    template <typename Extents>
    struct mapping : stdex::layout_right::mapping<Extents> {
      using base = stdex::layout_right::mapping<Extents>;
      using base::base;
    };
  };
};

struct struct_layout {

  template <typename Element_type>
  static constexpr auto default_access =
      [](Element_type &value) -> Element_type & { return value; };

  template <typename Element_type,
            typename Callable =
                std::remove_cv_t<decltype(default_access<Element_type>)>>
  struct struct_accessor {

    using offset_policy = stdex::default_accessor<Element_type>;
    using element_type = details::function_traits<Callable>::result_type;
    using reference = std::conditional_t<details::is_tuple<element_type>,
                                         element_type, element_type &>;
    using data_handle_type = Element_type *;

    constexpr struct_accessor() : proj(default_access<Element_type>){};
    explicit struct_accessor(Callable &&c) : proj(c) {}

    constexpr reference access(data_handle_type p, size_t i) const noexcept {
      if constexpr (std::is_rvalue_reference_v<reference>)
        return (proj(std::move(p[i])));
      else
        return proj(p[i]);
    }

  private:
    Callable proj;
  };
};

template <typename S, typename Callable>
auto make_struct_accessor(Callable &&c) {
  return struct_layout::struct_accessor<S, Callable>(std::forward<Callable>(c));
}

struct tiled_layout {

  // Hardwritten for 2D layouts, will change later...
  [[nodiscard]] static constexpr auto
  build(std::ranges::forward_range auto &&view, const size_t row,
        const size_t col) {
    return tiled_layout::build(
        view, stdex::dextents<size_t, 2>(row, col),
        stdex::default_accessor<std::ranges::range_value_t<decltype(view)>>());
  }

  // Hardwritten for 2D layouts, will change later...
  template <template <typename, size_t...> typename Extents, typename T,
            size_t... idx, typename Accessor>
  [[nodiscard]] static constexpr auto
  build(std::ranges::forward_range auto &&view, Extents<T, idx...> extents,
        const Accessor &acc) {
    // TODO: Check sizes against view
    // TODO: Tiled layout should work on every dimension
    using extent_type = Extents<T, idx...>;
    static_assert(extent_type::rank() == 2);

    tiled_layout_impl::mapping<extent_type> tiled_mapping(extents);
    using view_data_type = std::ranges::range_value_t<decltype(view)>;
    return stdex::mdspan<T, extent_type, tiled_layout_impl, Accessor>(
        std::ranges::data(view), tiled_mapping, acc);
  }

  struct tiled_layout_impl {

    template <class Extents> struct mapping {

      // for simplicity
      static_assert(Extents::rank() == 2,
                    "SimpleTileLayout2D is hard-coded for 2D layout");

      using extents_type = Extents;
      using rank_type = typename Extents::rank_type;
      using size_type = typename Extents::size_type;
      using layout_type = tiled_layout;

      mapping() noexcept = default;
      mapping(const mapping &) noexcept = default;
      mapping &operator=(const mapping &) noexcept = default;

      mapping(Extents const &exts, size_type row_tile,
              size_type col_tile) noexcept
          : extents_(exts), row_tile_size_(row_tile), col_tile_size_(col_tile) {
        // For simplicity, don't worry about negatives/zeros/etc.
        assert(row_tile > 0);
        assert(col_tile > 0);
        assert(exts.extent(0) > 0);
        assert(exts.extent(1) > 0);
      }

      MDSPAN_TEMPLATE_REQUIRES(
          class OtherExtents,
          /* requires */ (
              ::std::is_constructible<extents_type, OtherExtents>::value))
      MDSPAN_CONDITIONAL_EXPLICIT(
          (!::std::is_convertible<OtherExtents, extents_type>::value))
      constexpr mapping(const mapping<OtherExtents> &input_mapping) noexcept
          : extents_(input_mapping.extents()),
            row_tile_size_(input_mapping.row_tile_size_),
            col_tile_size_(input_mapping.col_tile_size_) {}

      //------------------------------------------------------------
      // Helper members (not part of the layout concept)

      constexpr size_type n_row_tiles() const noexcept {
        return extents_.extent(0) / row_tile_size_ +
               size_type((extents_.extent(0) % row_tile_size_) != 0);
      }

      constexpr size_type n_column_tiles() const noexcept {
        return extents_.extent(1) / col_tile_size_ +
               size_type((extents_.extent(1) % col_tile_size_) != 0);
      }

      constexpr size_type tile_size() const noexcept {
        return row_tile_size_ * col_tile_size_;
      }

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

      template <class RowIndex, class ColIndex>
      // requires(is_convertible_v<RowIndex, size_type> &&
      //   is_convertible_v<ColIndex, size_type> &&
      //   is_nothrow_constructible_v<size_type, RowIndex> &&
      //   is_nothrow_constructible_v<size_type, ColIndex>)
      constexpr size_type operator()(RowIndex row,
                                     ColIndex col) const noexcept {
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
        return (extents_.extent(0) % row_tile_size_ == 0) &&
               (extents_.extent(1) % col_tile_size_ == 0);
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

  struct block_layout_impl {

    template <typename Extents> struct mapping {

      static_assert(Extents::rank() == 1,
                    "Hardwritten for 1D layouts, for now");

      using extents_type = Extents;
      using rank_type = typename Extents::rank_type;
      using size_type = typename Extents::size_type;
      using layout_type = block_layout::block_layout_impl;

      mapping() noexcept = default;
      mapping(const mapping &) noexcept = default;
      mapping &operator=(const mapping &) noexcept = default;

      template <details::typed_range<size_t> K>
      constexpr mapping(const Extents& ext, K &&b, K &&s)
          : _extents(ext), blocks(b), strides(s), sum_blocks(0), sum_diffs(0)
      {
        // At least one block and stride
        for (int i = 0; i < blocks.size(); i++) {
          diffs.push_back(strides[i] - blocks[i]);
          sum_blocks += blocks[i];
          sum_diffs += diffs[i];
        }

        offsets.resize(diffs.size() - 1);
        std::inclusive_scan(diffs.begin(), diffs.end() -1 ,offsets.begin());

        partial_blocks.resize(blocks.size());
        std::inclusive_scan(blocks.begin(), blocks.end() ,partial_blocks.begin());
      }

      // TODO: copy constructor

      // Mandatory member methods

      constexpr const extents_type &extents() const { return _extents; }

      constexpr size_type required_span_size() const noexcept {
        // return n_row_tiles() * n_column_tiles() * tile_size();
      }

      template<class index>
      constexpr inline size_t pos_in_blocks(index idx) const noexcept {
        return idx / sum_blocks;
      }

      template<class index>
      constexpr inline size_t offset_in_block(index idx) const noexcept {
        const auto pos = idx % sum_blocks;
        if(pos <= blocks[0])
            return 0;
        
        const auto new_pos = std::distance(partial_blocks.begin(),
                                            std::find_if(partial_blocks.begin(), partial_blocks.end(), [pos](auto& e){return pos < e;})) - 1;
        return offsets[new_pos];
      }

      template <class index>
      constexpr size_type operator()(index idx) const noexcept {
        return idx + sum_diffs * pos_in_blocks(idx) + offset_in_block(idx);  
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
        return false; //TODO....
      }
      // There are some circumstances where this is strided, but we're not
      // concerned about that optimization, so we're allowed to just return
      // false here
      constexpr bool is_strided() const noexcept { return true; }

    private:
      Extents _extents;
      std::vector<size_t> blocks;
      std::vector<size_t> partial_blocks;
      std::vector<size_t> strides;
      std::vector<size_t> diffs;
      std::vector<size_t> offsets;
      size_t sum_diffs{};
      size_t sum_blocks{};
    };
  };
};

} // namespace empi::layouts

#endif /* INCLUDE_EMPI_LAYOUTS */
