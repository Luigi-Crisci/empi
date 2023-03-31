//
// Created by luigi on 21/09/22.
//

#ifndef EMPI_PROJECT_UTILS_HPP
#define EMPI_PROJECT_UTILS_HPP

#include <empi/type_traits.hpp>
#include <empi/defines.hpp>
#include <iostream>

namespace empi::details{


  template<typename T>
  struct conditional_deleter{
    
    constexpr conditional_deleter() noexcept = default;
    constexpr explicit conditional_deleter(bool owving) : _is_owning_ptr(owving) {}

    template<typename Up>
    requires std::is_convertible_v<Up*, T*>
    constexpr explicit conditional_deleter(const conditional_deleter<Up>& cd) noexcept {}

    void operator()(T* _ptr) const {
      if(_is_owning_ptr){
        delete _ptr;
	  }
      #ifdef TEST
      _ptr = nullptr;
      #endif
    }

    private:
      bool _is_owning_ptr = false;
  };

    template<typename T>
    struct conditional_deleter<T[]>{
    
    constexpr conditional_deleter() noexcept = default;
    constexpr explicit conditional_deleter(bool owving) : _is_owning_ptr(owving) {}

    template<typename Up>
    requires std::is_convertible_v<Up(*)[], T(*)[]>
    constexpr explicit conditional_deleter(const conditional_deleter<Up[]>& cd) noexcept {}

    template<typename Up>
    requires std::is_convertible_v<Up(*)[], T(*)[]>
    void operator()(Up* _ptr) const {
      if(_is_owning_ptr)
        delete[] _ptr;
      #ifdef TEST
      _ptr = nullptr;
      #endif
    }

    private:
      bool _is_owning_ptr = false;
  };



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

//////////////////////// Struct layout lambda helper ///////////////////////////////
#define PARENS ()

#define EXPAND(...) EXPAND4(EXPAND4(EXPAND4(EXPAND4(__VA_ARGS__))))
#define EXPAND4(...) EXPAND3(EXPAND3(EXPAND3(EXPAND3(__VA_ARGS__))))
#define EXPAND3(...) EXPAND2(EXPAND2(EXPAND2(EXPAND2(__VA_ARGS__))))
#define EXPAND2(...) EXPAND1(EXPAND1(EXPAND1(EXPAND1(__VA_ARGS__))))
#define EXPAND1(...) __VA_ARGS__

#define FOR_EACH(macro, _struct, ...)                                    \
  __VA_OPT__(EXPAND(FOR_EACH_HELPER(macro,_struct, __VA_ARGS__)))
#define FOR_EACH_HELPER(macro, _struct, a1, ...)                         \
  macro(_struct, a1)                                                     \
  __VA_OPT__(,FOR_EACH_AGAIN PARENS (macro, _struct, __VA_ARGS__))
#define FOR_EACH_AGAIN() FOR_EACH_HELPER

#define F(_struct, field) _struct.field

#define STRUCT_FIELDS(_struct, ...) \
    [](_struct& s) -> auto {return std::tuple{FOR_EACH(F,s,__VA_ARGS__)};}

//////////////////////////////////
		
}

#endif //EMPI_PROJECT_UTILS_HPP
