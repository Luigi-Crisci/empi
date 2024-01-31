# 
# Copyright (c) 2022-2023 University of Salerno, Italy. All rights reserved.
# 

# Find catch_discore_tests
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/lib/Catch2/contrib)

include(CTest)
include(Catch)

# Link examples to Catch2Main
macro(add_test_executable target_name source)
	add_executable(${target_name} ${source})
	target_compile_definitions(${target_name} PUBLIC -DTEST)
	target_link_libraries(${target_name} Catch2::Catch2WithMain)
	#Disable char* string conversion warning
	target_compile_options(${target_name} PUBLIC "-Wno-write-strings") 
	catch_discover_tests(${target_name})	
endmacro()

add_test_executable(test_layouts test_layouts.cpp)
add_test_executable(test_type_traits test_type_traits.cpp)
add_test_executable(test_message_group_hdl test_message_group_hdl.cpp)
add_test_executable(test_utils test_utils.cpp)
add_test_executable(test_compact test_compact.cpp)

