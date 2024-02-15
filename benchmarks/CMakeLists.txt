# Copyright (c) 2022-2023 University of Salerno, Italy. All rights reserved.

# fetch latest argparse
include(FetchContent)
FetchContent_Declare(argparse
                     GIT_REPOSITORY https://github.com/p-ranav/argparse.git)
FetchContent_MakeAvailable(argparse)

macro(create_example TARGET_NAME SRC_FILE)
  add_executable(${TARGET_NAME} ${SRC_FILE})
  set(EMPI_BENCHMARK_NAME ${TARGET_NAME})
  # Check if target name contains pack substring
  string(FIND ${TARGET_NAME} "pack" PACK_FOUND)
  if(PACK_FOUND GREATER -1)
    set(PACK_FOUND TRUE)
  else()
    set(PACK_FOUND FALSE)
  endif()
  target_compile_definitions(${TARGET_NAME} PRIVATE $<$<BOOL:${PACK_FOUND}>:EMPI_INCLUDE_UNPACK_TIME>)
  target_compile_definitions(${TARGET_NAME}
                             PRIVATE EMPI_BENCHMARK_NAME="${TARGET_NAME}")
  target_link_libraries(${TARGET_NAME} empi argparse)
endmacro()

option(BUILD_MPI_EXAMPLES "Build MPI examples" OFF)
option(BUILD_UMPI_EXAMPLES "Build UMPI examples" OFF)
option(BUILD_MPL_EXAMPLES "Build MPL examples" OFF)

add_subdirectory(all_reduce)
add_subdirectory(bcast)
add_subdirectory(bdring)
add_subdirectory(ibcast)
add_subdirectory(ping_pong)
add_subdirectory(vibrating_string)

add_subdirectory(datatypes)
