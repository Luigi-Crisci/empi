#ifndef INCLUDE_EMPI_LAYOUTS
#define INCLUDE_EMPI_LAYOUTS

#include <functional>
#include <memory>
#include <cassert>
#include <utility>

#include <empi/datatype.hpp>
#include <empi/defines.hpp>

namespace empi::layouts{

	template<typename T, typename Extents, typename AccessPolicy = stdex::default_accessor<T>>
	using column_view = stdex::mdspan<T, Extents,stdex::layout_stride,AccessPolicy>;

	struct column_layout {
		
		//Hardwritten for 2D layouts, will change later...
		template<template<typename, size_t...> typename Extents, typename K, size_t ...Idx>
		[[nodiscard]] static auto build(std::ranges::forward_range auto&& view, Extents<K,Idx...> extents, size_t col){
			return column_layout::build(view, extents, col, stdex::default_accessor<std::ranges::range_value_t<decltype(view)>>());
		}

		//Hardwritten for 2D layouts, will change later...
		template<template<typename, size_t...> typename Extents, typename K, size_t ...Idx, typename Accessor>
		[[nodiscard]] static auto build(std::ranges::forward_range auto&& view, Extents<K,Idx...> extents, size_t col, const Accessor& acc){
			static_assert(Extents<K,Idx...>::rank() == 2);
			assert(col < extents.extent(1));
			using extent_type = decltype(details::remove_last(extents));
			extent_type new_extents(extents.extent(0));
			std::array stride{extents.extent(1)};

			column_layout_impl::mapping <extent_type> column_map(new_extents, stride);

			using T = std::ranges::range_value_t<decltype(view)>;

			return stdex::mdspan<T, extent_type, column_layout_impl, Accessor>(
				std::ranges::data(view) + col,
				column_map,
				acc);
		}

		struct column_layout_impl {
			
			template<typename Extents>
			struct mapping : stdex::layout_stride::mapping<Extents> {
				using base = stdex::layout_stride::mapping<Extents>;
				using base::base;
			};
		};
	};

	struct contiguous_layout{
		// Wraps data into plain, 1D mdspan.
		// Used for implicit conversions
		[[nodiscard]] static auto build(std::ranges::forward_range auto&& view){
			using extent_type = stdex::dextents<std::size_t, 1>;
			extent_type extents(std::ranges::size(view));
			using T = std::ranges::range_value_t<decltype(view)>;
			return std::move(stdex::mdspan<T, extent_type, contiguous_layout_impl>(std::ranges::data(view),
																							extents));
		}

		template<typename Accessor>
		[[nodiscard]] static auto build(std::ranges::forward_range auto&& view, const Accessor& acc){
			using extent_type = stdex::dextents<std::size_t, 1>;
			extent_type extents(std::ranges::size(view));
			using T = std::ranges::range_value_t<decltype(view)>;
			return std::move(stdex::mdspan<T, extent_type, contiguous_layout_impl,Accessor>(std::ranges::data(view),
																				   contiguous_layout_impl::mapping<extent_type>(extents),
																				   acc));
		}

		struct contiguous_layout_impl {
			template<typename Extents>
			struct mapping : stdex::layout_right::mapping<Extents> {
				using base = stdex::layout_right::mapping<Extents>;
				using base::base;
			};
		};
	};

	struct struct_layout {

		template<typename Element_type>
		static constexpr auto default_access = [](Element_type& value) -> Element_type& {return value;};

		template<typename Element_type, typename Callable = std::remove_cv_t<decltype(default_access<Element_type>)> >
		struct struct_accessor{

			using offset_policy = stdex::default_accessor<Element_type>;
			using element_type = details::function_traits<Callable>::result_type;
			using reference = std::conditional_t<details::is_tuple<element_type>,element_type, element_type&>;
			using data_handle_type = Element_type*;
			
			constexpr struct_accessor() : proj(default_access<Element_type>) {};
			explicit struct_accessor(Callable&& c) : proj(c) {}

			constexpr reference access(data_handle_type p, size_t i) const noexcept{
					if constexpr (std::is_rvalue_reference_v<reference>)
						return (proj(std::move(p[i])));
					else
						return proj(p[i]);
			}

			private:
				Callable proj;
		};
			
	};
	

	template<typename S, typename Callable>
	auto make_struct_accessor(Callable&& c){
		return struct_layout::struct_accessor<S, Callable>(std::forward<Callable>(c));
	}





}

#endif /* INCLUDE_EMPI_LAYOUTS */
