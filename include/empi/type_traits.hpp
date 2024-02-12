/*
 * Copyright (c) 2022-2023 University of Salerno, Italy. All rights reserved.
 */

#ifndef __TYPE_TRAITS_H__
#define __TYPE_TRAITS_H__

#include <mpi.h>

#include <mdspan/mdspan.hpp>
#include <tuple>
#include <type_traits>

namespace empi::details {

template<typename T, T V1, T V2>
struct is_greater {
    static constexpr bool value = V1 > V2;
};

template<typename T, T V1, T V2>
struct is_same {
    static constexpr bool value = V1 == V2;
};

template<std::size_t Size>
concept has_size = requires { Size > 0; };


template<typename T, T V1, T V2>
constexpr bool is_greater_v = is_greater<T, V1, V2>::value;
template<typename T, T V1, T V2>
constexpr bool is_same_v = is_same<T, V1, V2>::value;


// https://stackoverflow.com/a/7943765
template<typename T>
struct function_traits : public function_traits<decltype(&T::operator())> {};
// For generic types, directly use the result of the signature of its 'operator()'

template<typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType (ClassType::*)(Args...) const>
// we specialize for pointers to member function
{
    enum { arity = sizeof...(Args) };
    // arity is the number of arguments.

    typedef ReturnType result_type;

    template<std::size_t I>
    struct arg {
        typedef typename std::tuple_element<I, std::tuple<Args...>>::type type;
        // the i-th argument is equivalent to the i-th tuple element of a tuple
        // composed of those arguments.
    };
};

template<class, template<class, class...> class>
struct is_instance : public std::false_type {};

template<class... Ts, template<class, class...> class U>
struct is_instance<U<Ts...>, U> : public std::true_type {};


template<typename T, template<class, class...> class K>
concept has_parameter = requires(
    T f) { is_instance<std::remove_reference_t<typename function_traits<decltype(f)>::template arg<0>::type>, K>{}; };

template<class T, class U = std::remove_cvref_t<std::remove_pointer_t<std::remove_extent_t<T>>>>
struct remove_all : remove_all<U> {};
template<class T>
struct remove_all<T, T> {
    typedef T type;
};
template<class T>
using remove_all_t = typename remove_all<T>::type;


// ----------- CONCEPTS ------------
template<typename T>
concept has_data = requires(T t) {
    { t.data() };
};

template<typename T>
concept has_value_type = requires(T t) { typename T::value_type; };

template<typename T>
concept has_accessor_type = requires(T t) { typename T::accessor_type; };


template<typename T, typename K>
concept is_valid_container = has_data<K> && std::is_same_v<T, typename std::remove_reference_t<K>::value_type>;

template<typename T, typename K>
concept is_valid_pointer = std::is_same_v<T, remove_all_t<K>>;


template<template<typename IndexType, size_t... Args> typename T, typename IndexType, size_t FirstEntryFrom,
    size_t... TupleTypesResult, size_t... TupleTypesFrom>
constexpr auto remove_last_impl(
    T<IndexType, TupleTypesResult...> res, T<IndexType, FirstEntryFrom, TupleTypesFrom...> src) {
    if constexpr(sizeof...(TupleTypesFrom) > 0) {
        return remove_last_impl(T<IndexType, TupleTypesResult..., FirstEntryFrom>(), T<IndexType, TupleTypesFrom...>());
    } else {
        return res;
    }
}

template<template<typename IndexType, size_t... Args> typename T, typename IndexType, size_t... Args>
constexpr auto remove_last(T<IndexType, Args...> in) {
    return remove_last_impl(T<IndexType>(), in);
}

template<typename A, typename B>
struct is_same_template : std::false_type {};

template<template<typename...> typename Template, typename... Args, typename... Brgs>
struct is_same_template<Template<Args...>, Template<Brgs...>> : std::true_type {};

template<typename T, typename K>
static constexpr bool is_same_template_v = is_same_template<T, K>::value;

// Simple and incomplete concept to check if a type has static extent
template<typename C>
concept has_static_extent
    = requires(C c) { (c.extent != Kokkos::dynamic_extent) || (std::is_array_v<C>) || (std::extent_v<C> > 0); };

///////////////// IS MDSPAN ////////////////

template<typename T>
struct is_mdspan_impl : std::false_type {};

template<typename T, typename Extents, typename Layout, typename Accessor>
struct is_mdspan_impl<Kokkos::mdspan<T, Extents, Layout, Accessor>> : std::true_type {};

template<typename T>
static constexpr bool is_mdspan_v = is_mdspan_impl<T>::value;

template<typename T>
concept is_mdspan = is_mdspan_v<std::remove_cvref_t<T>>;

///////////////// size_of //////////////

template<typename T>
constexpr size_t size_of = sizeof(T);

template<typename T, typename... Args>
constexpr size_t size_of<std::tuple<T, Args...>> = sizeof(T) + (sizeof(Args) + ...);

////////////////////////// Is tuple //////////////////

template<typename T>
struct is_tuple_impl : std::false_type {};

template<typename T, typename... Args>
struct is_tuple_impl<std::tuple<T, Args...>> : std::true_type {};

template<typename T>
static constexpr bool is_tuple_v = is_tuple_impl<T>::value;

template<typename T>
concept is_tuple = is_tuple_v<std::remove_cvref_t<T>>;

//////////////////////// Get type //////////////////////
template<typename T>
struct get_true_type {
    using type = T;
};

template<typename T>
    requires has_value_type<T> && (!has_accessor_type<T>)
struct get_true_type<T> {
    using type = typename T::value_type;
};

template<is_mdspan T>
struct get_true_type<T> {
    using type = typename details::remove_all_t<T>::accessor_type::element_type;
};

template<typename T>
    requires std::is_pointer_v<T>
struct get_true_type<T> {
    using type = std::remove_pointer_t<T>;
};

template<typename T>
using get_true_type_t = typename get_true_type<remove_all_t<T>>::type;

//////////////// Typed range //////////////
template<typename Range, typename Type>
concept typed_range = std::ranges::range<Range> && std::is_same_v<std::ranges::range_value_t<Range>, Type>;

} // namespace empi::details


#endif // __TYPE_TRAITS_H__