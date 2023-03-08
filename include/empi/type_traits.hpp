#ifndef __TYPE_TRAITS_H__
#define __TYPE_TRAITS_H__

#include <mpi.h>

#include <type_traits>
#include <tuple>
#include <concepts>
#include <experimental/mdspan>

namespace empi
{
namespace details {

template<typename T, T v1, T v2>
struct is_greater {
  static constexpr bool value = v1 > v2;
};

template<typename T, T v1, T v2>
struct is_same {
  static constexpr bool value = v1 == v2;
};

template<std::size_t size>
concept has_size = requires {
  size > 0;
};


template<typename T, T v1, T v2>
constexpr bool is_greater_v = is_greater<T,v1,v2>::value;
template<typename T, T v1, T v2>
constexpr bool is_same_v = is_same<T,v1,v2>::value;

}

// https://stackoverflow.com/a/7943765
template <typename T>
struct function_traits
	: public function_traits<decltype(&T::operator())>
{};
// For generic types, directly use the result of the signature of its 'operator()'

template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType(ClassType::*)(Args...) const>
// we specialize for pointers to member function
{
  enum { arity = sizeof...(Args) };
  // arity is the number of arguments.

  typedef ReturnType result_type;

  template <std::size_t i>
  struct arg
  {
	typedef typename std::tuple_element<i, std::tuple<Args...>>::type type;
	// the i-th argument is equivalent to the i-th tuple element of a tuple
	// composed of those arguments.
  };
};

template <class, template <class, class...> class>
struct is_instance : public std::false_type {};

template <class...Ts, template <class, class...> class U>
struct is_instance<U<Ts...>, U> : public std::true_type {};


template<typename T, template <class, class...> class K>
concept has_parameter = requires(T f){
	is_instance<std::remove_reference_t<typename function_traits<decltype(f)>::template arg<0>::type>,K>{};
};

template<class T, class U=
std::remove_cvref_t<
	std::remove_pointer_t<
		std::remove_extent_t<
			T >>>>
struct remove_all : remove_all<U> {};
template<class T> struct remove_all<T, T> { typedef T type; };
template<class T> using remove_all_t = typename remove_all<T>::type;


// ----------- CONCEPTS ------------
template<typename T>
concept has_data = requires(T t){
  {t.data()};
};

template<typename T>
concept has_value_type = requires(T t){
  typename T::value_type;
};

template<typename T, typename K>
concept is_valid_container = has_data<K> && std::is_same_v<T, typename std::remove_reference_t<K>::value_type>;

template<typename T, typename K>
concept is_valid_pointer = std::is_same_v<T,remove_all_t<K>>;


template<typename T>
struct get_true_type{
  using type = T;
};

template<has_value_type T>
struct get_true_type<T>{
  using type = typename T::value_type;
};

template<template<typename index_type,size_t ...Args> typename T,typename index_type, size_t FirstEntryFrom, size_t ...TupleTypesResult, size_t ...TupleTypesFrom>
constexpr auto remove_last_impl(T<index_type,TupleTypesResult...> res, T<index_type, FirstEntryFrom, TupleTypesFrom...> src){
    if constexpr(sizeof...(TupleTypesFrom) > 0)
        return remove_last_impl(T<index_type, TupleTypesResult..., FirstEntryFrom>(), T<index_type, TupleTypesFrom...>());
    else 
        return res;
}

template<template<typename index_type,size_t ...Args> typename T, typename index_type, size_t ...Args>
constexpr auto remove_last(T<index_type,Args...> in){
    return remove_last_impl(T<index_type>(), in);
}

template<typename A, typename B>
struct is_same_template : std::false_type {};

template<template<typename...> typename Template, typename... Args, typename... Brgs> 
struct is_same_template<Template<Args...>,Template<Brgs...>> : std::true_type {};

template<typename T, typename K>
static constexpr bool is_same_template_v = is_same_template<T, K>::value;

template<typename T>
concept is_mdspan = requires{ is_same_template_v<T, std::experimental::mdspan<int, std::experimental::dextents<int, 1>>>;};




} // namespace empi


#endif // __TYPE_TRAITS_H__