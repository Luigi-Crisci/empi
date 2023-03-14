
# Find catch_discore_tests
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/lib/Catch2/contrib)

include(CTest)
include(Catch)

# Cache Catch2 compilation by creating a static library
# add_library(Catch2Main STATIC static_main.cpp)
# target_link_libraries(Catch2Main Catch2::Catch2)

# Link examples to Catch2Main
macro(add_test_executable target_name source)
	add_executable(${target_name} ${source})
	target_link_libraries(${target_name} Catch2::Catch2WithMain)
	catch_discover_tests(${target_name})	
endmacro()

add_test_executable(test_layouts test_layouts.cpp)
add_test_executable(test_type_traits test_type_traits.cpp)
add_test_executable(test_message_group_hdl test_message_group_hdl.cpp)

