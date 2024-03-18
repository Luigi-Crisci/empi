#
# Copyright (c) 2022-2023 University of Salerno, Italy. All rights reserved.
#

cmake_minimum_required(VERSION 3.0.0)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
  cmake_policy(SET CMP0135 NEW) # ExternalProject downloads will have archive
                                # timestamps
endif()

set(EMPI_VERSION 0.1.0)
set(EMPI_VERSION_MAJOR 0)
set(EMPI_VERSION_MINOR 1)
set(EMPI_VERSION_PATCH 0)

project(
  empi
  VERSION "${EMPI_VERSION}"
  LANGUAGES CXX)

add_library(empi INTERFACE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)
# fetch mdspan
include(FetchContent)
FetchContent_Declare(
  mdspan
  GIT_REPOSITORY https://github.com/kokkos/mdspan.git
  GIT_TAG stable)
FetchContent_MakeAvailable(mdspan)

include(FindMPI REQUIRED)

option(ENABLE_UNCHECKED_FUNCTION
       "Enable unchecked function (requires custom OMPI)" ON)
option(BUILD_BENCHMARKS "Build Benchmarks" OFF)
option(ENABLE_TESTS "Enable testing" OFF)
option(ENABLE_SANITIZER "Enable sanitizer" OFF)

# Default build flags
set(CMAKE_CXX_FLAGS_DEBUG
    "-O0 -g -DDEBUG -fno-omit-frame-pointer"
    CACHE STRING "Flags used by the C++ compiler during debug builds." FORCE)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
    "-O2 -DNDEBUG -march=native -ffast-math -g -fno-omit-frame-pointer"
    CACHE
      STRING
      "Flags used by the C++ compiler during release builds with debug info."
      FORCE)
set(CMAKE_CXX_FLAGS_RELEASE
    "-O3 -DNDEBUG -march=native -ffast-math"
    CACHE STRING "Flags used by the C++ compiler during release builds." FORCE)

if(ENABLE_SANITIZER)
  set(CMAKE_CXX_FLAGS_DEBUG
      "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address,undefined,leak")
endif()

set(CONFIG_PATH "${CMAKE_CURRENT_BINARY_DIR}/include/empi/config.hpp")
configure_file("${PROJECT_SOURCE_DIR}/include/empi/config.hpp.in"
               ${CONFIG_PATH} @ONLY)

# Force colorful diagnostic
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  add_compile_options(-fdiagnostics-color=always)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  add_compile_options(-fcolor-diagnostics)
endif()

set(header_files
    ${CMAKE_CURRENT_SOURCE_DIR}/include/empi/empi.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/empi/context.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/empi/type_traits.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/empi/message_grp_hdl.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/empi/tag.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/empi/utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/empi/message_group.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/empi/request_pool.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/empi/async_event.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/empi/datatype.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/empi/defines.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/empi/layouts.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/empi/layouts_traits.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/empi/time.hpp
    # Compact files
    ${CMAKE_CURRENT_SOURCE_DIR}/include/empi/compact/compact.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/empi/compact/compact_layout_stride.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/empi/compact/compact_common.hpp

    ${CONFIG_PATH})

get_property(
  dirs1
  TARGET std::mdspan
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES)

target_include_directories(
  empi
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> ${dirs1})

target_link_libraries(empi INTERFACE MPI::MPI_CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE
      "Release"
      CACHE STRING "The type of build" FORCE)
  message(
    STATUS "Setting build type to '${CMAKE_BUILD_TYPE}' as none was specified")
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
                                               "RelWithDebInfo")
endif()

# Force color diagnostics on Ninja
if(CMAKE_GENERATOR STREQUAL "Ninja")
  target_compile_options(empi INTERFACE -fdiagnostics-color=always)
endif()

if(ENABLE_TESTS)
  include(CTest)
  enable_testing()
  add_subdirectory(lib/Catch2) # Find Catch2
  add_subdirectory(test)
endif()

if(BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# Install stuffs

install(
  TARGETS empi
  EXPORT ${PROJECT_NAME}_Targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "empiConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

install(
  EXPORT ${PROJECT_NAME}_Targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
              "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/empi DESTINATION include)

message("//===---------------------------------------===//")
message("//           CONFIGURATION SUMMARY             //")
message("//===---------------------------------------===//")
message("-> CMAKE_BUILD_TYPE:.................. ${CMAKE_BUILD_TYPE}")
message("-> CMAKE_INSTALL_PREFIX:.............. ${CMAKE_INSTALL_PREFIX}")
message("-> CMAKE_CXX_COMPILER:................ ${CMAKE_CXX_COMPILER}")
message("-> ENABLE_UNCHECKED_FUNCTION:........ ${ENABLE_UNCHECKED_FUNCTION}")
message("-> BUILD_BENCHMARKS:................... ${BUILD_BENCHMARKS}")
if(BUILD_BENCHMARKS)
  message("---> BUILD_MPI_EXAMPLES:............. ${BUILD_MPI_EXAMPLES}")
  message("---> BUILD_UMPI_EXAMPLES:............ ${BUILD_UMPI_EXAMPLES}")
  message("---> BUILD_MPL_EXAMPLES:............. ${BUILD_MPL_EXAMPLES}")
endif()
message("-> ENABLE_TESTS:..................... ${ENABLE_TESTS}")
message("//===---------------------------------------===//")
