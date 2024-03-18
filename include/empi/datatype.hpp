/*
 * Copyright (c) 2022-2023 University of Salerno, Italy. All rights reserved.
 */

#ifndef EMPI_PROJECT_INCLUDE_EMPI_DATATYPE_HPP_
#define EMPI_PROJECT_INCLUDE_EMPI_DATATYPE_HPP_

//@HEADER
#include <mdspan/mdspan.hpp>

#include <empi/type_traits.hpp>
#include <empi/utils.hpp>
#include <memory>


namespace empi::details {

static constexpr bool no_status = false;

template<typename T>
struct mpi_type_impl {
    static MPI_Datatype get_type() noexcept { return MPI_DATATYPE_NULL; }
};

#define MAKE_TYPE_CONVERSION(T, base_type)                                                                             \
    template<>                                                                                                         \
    struct mpi_type_impl<T> {                                                                                          \
        static MPI_Datatype get_type() noexcept { return base_type; }                                                  \
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
        if constexpr(has_data<T>) {
            return mpi_type_impl<typename T::value_type>::get_type();
        } else {
            return mpi_type_impl<std::remove_pointer_t<T>>::get_type();
        }
    }
};


#define empi_byte_cast(ptr) (static_cast<std::byte *>(static_cast<void *>((ptr).get())))

template<typename T>
    requires has_data<T>
static inline constexpr auto get_underlying_pointer(T &&buf) {
    using element_type = typename std::remove_reference_t<T>::value_type;
    return std::unique_ptr<element_type, empi::details::conditional_deleter<element_type>>(buf.data());
}

template<typename T>
static inline constexpr auto get_underlying_pointer(T *buf) {
    return std::unique_ptr<T, empi::details::conditional_deleter<T>>(buf);
}

template<typename T>
static inline constexpr auto get_underlying_pointer(const T *buf) {
    return std::unique_ptr<const T, empi::details::conditional_deleter<const T>>(buf);
}

template<typename T>
    requires std::is_arithmetic_v<T>
static inline constexpr auto get_underlying_pointer(T &buf) {
    return std::unique_ptr<T, empi::details::conditional_deleter<T>>(&buf);
}

template<typename T>
    requires std::is_arithmetic_v<T>
static inline constexpr auto get_underlying_pointer(const T &buf) {
    return std::unique_ptr<const T, empi::details::conditional_deleter<T>>(&buf);
}

template<typename T>
static inline constexpr auto get_underlying_pointer(std::unique_ptr<T> &buf) {
    return std::move(buf);
}

template<typename T>
    requires(!Mdspan<T>)
static inline constexpr auto get_underlying_pointer(T &&buf, bool flag) {
    return get_underlying_pointer(std::forward<T>(buf));
}


} // namespace empi::details

#endif // EMPI_PROJECT_INCLUDE_EMPI_DATATYPE_HPP_
