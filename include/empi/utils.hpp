//
// Created by luigi on 21/09/22.
//

#ifndef EMPI_PROJECT_UTILS_HPP
#define EMPI_PROJECT_UTILS_HPP

#include <empi/type_traits.hpp>
#include <empi/defines.hpp>

namespace empi::details{

		template<typename T>
		requires has_data<T>
		static inline auto get_underlying_pointer(T&& buf){
		  return buf.data();
		}

		template<typename T>
		static inline auto get_underlying_pointer(T* buf){
		  return buf;
		}

		template<typename T>
		static inline auto get_underlying_pointer(const T* buf){
		  return buf;
		}
		
		template<typename T>
		inline constexpr auto abs(T& a, T&b) -> decltype(std::declval<T>() - std::declval<T>()) {
			return std::abs(static_cast<long long>(a) - static_cast<long long>(b));
		}

		template<mpi_function f> 
		void checktag(int tag, int maxtag){
			if constexpr (details::is_all<f>){
				if(tag > maxtag)
					throw std::runtime_error("Incorrect tag entered in send function");
			}
			else if constexpr (details::is_send<f>){
				if(tag > maxtag || tag == -1)
					throw std::runtime_error("Incorrect tag entered in send function");
			}
			else if constexpr (details::is_recv<f>){
				if(tag > maxtag || tag < -1)
					throw std::runtime_error("Incorrect tag entered in recv function");
			}
		}

		template<is_mdspan T, typename Op>
		requires (T::extents_type::rank() == 1)
		void constexpr apply(T view, Op&& op){
			for(int i = 0; i < view.extent(0); i++){
				op(view[i]);
			}
		}

		template<is_mdspan T, typename Op>
		requires (T::extents_type::rank() == 2)
		void constexpr apply(T view, Op&& op){
			for(int i = 0; i < view.extent(0); i++){
				for (int j = 0; j < view.extent(1); j++) {
					op(view(i,j));
				}
			}
		}

		template<is_mdspan T, typename Op>
		requires (T::extents_type::rank() == 3)
		void constexpr apply(T view, Op&& op){
			for(int i = 0; i < view.extent(0); i++){
				for (int j = 0; j < view.extent(1); j++) {
					for (int z = 0; i < view.extent(2); z++){
						op(view(i,j,z));
					}
				}
			}
		}


		
}

#endif //EMPI_PROJECT_UTILS_HPP
