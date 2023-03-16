//
// Created by luigi on 05/10/22.
//

#ifndef EMPI_PROJECT_INCLUDE_EMPI_DATATYPE_HPP_
#define EMPI_PROJECT_INCLUDE_EMPI_DATATYPE_HPP_

#include <type_traits>
#include <memory.h>

#include <experimental/mdspan>
#include <empi/type_traits.hpp>


namespace empi::details {

static constexpr bool no_status = false;


template<typename T>
struct mpi_type_impl {
  static MPI_Datatype get_type() noexcept { return nullptr; }
};

#define MAKE_TYPE_CONVERSION(T, base_type) \
    template<> \
    struct mpi_type_impl<T> \
    {                                      \
        static MPI_Datatype get_type() noexcept { return base_type; } \
    };

MAKE_TYPE_CONVERSION(int, MPI_INT)
MAKE_TYPE_CONVERSION(char, MPI_CHAR)
MAKE_TYPE_CONVERSION(short, MPI_SHORT)
MAKE_TYPE_CONVERSION(long, MPI_LONG)
MAKE_TYPE_CONVERSION(float, MPI_FLOAT)
MAKE_TYPE_CONVERSION(double, MPI_DOUBLE)

template<typename T>
struct mpi_type {
  static MPI_Datatype get_type() {
	if constexpr (has_data<T>)
	  return mpi_type_impl<typename T::value_type>::get_type();
	else
	  return mpi_type_impl<std::remove_pointer_t<T>>::get_type();
  }
};

#define empi_byte_cast(ptr) (static_cast<std::byte*>(static_cast<void*>(ptr)))

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
requires std::is_arithmetic_v<T>
static inline auto get_underlying_pointer(T& buf){
  return &buf;
}

template<typename T>
requires std::is_arithmetic_v<T>
static inline auto get_underlying_pointer(const T& buf){
  return &buf;
}

}

#endif //EMPI_PROJECT_INCLUDE_EMPI_DATATYPE_HPP_
