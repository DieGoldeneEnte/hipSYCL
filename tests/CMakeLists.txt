project(hipsycl-tests)
cmake_minimum_required(VERSION 3.5)

set(Boost_USE_STATIC_LIBS off)
set(BUILD_SHARED_LIBS on)
find_package(Boost COMPONENTS unit_test_framework REQUIRED)

find_package(hipSYCL REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT HIPSYCL_DEBUG_LEVEL)
  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(HIPSYCL_DEBUG_LEVEL 3 CACHE INTEGER
      "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
    FORCE)
  else()
    set(HIPSYCL_DEBUG_LEVEL 1 CACHE INTEGER
      "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
      FORCE)
  endif()
endif()

if(CMAKE_GENERATOR STREQUAL "Ninja" AND
    ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9) OR
    (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.5)))
  # Force colored warnings in Ninja's output, if the compiler has -fdiagnostics-color support.
  # Rationale in https://github.com/ninja-build/ninja/issues/814
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
endif()

#add_compile_definitions(HIPSYCL_DEBUG_LEVEL="${HIPSYCL_DEBUG_LEVEL}")
#Use add_definitions for now for older cmake versions
cmake_policy(SET CMP0005 NEW)
add_definitions(-DHIPSYCL_DEBUG_LEVEL=${HIPSYCL_DEBUG_LEVEL})
if(HIPSYCL_DISABLE_UNNAMED_LAMBDA_TESTS)
  add_definitions(-DHIPSYCL_DISABLE_UNNAMED_LAMBDA_TESTS)
endif()

add_subdirectory(platform_api)

add_subdirectory(dump_test)
add_executable(device_compilation_tests device_compilation_tests.cpp)
target_include_directories(device_compilation_tests PRIVATE ${Boost_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(device_compilation_tests PRIVATE ${Boost_LIBRARIES})
add_sycl_to_target(TARGET device_compilation_tests)

add_executable(sycl_tests
  sycl/smoke/task_graph.cpp
  sycl/accessor.cpp
  sycl/buffer.cpp
  sycl/explicit_copy.cpp
  sycl/extensions.cpp
  sycl/fill.cpp
  sycl/group_functions/group_functions_misc.cpp
  sycl/group_functions/group_functions_binary_reduce.cpp
  sycl/group_functions/group_functions_reduce.cpp
  sycl/group_functions/group_functions_scan.cpp
  sycl/id_range.cpp
  sycl/item.cpp
  sycl/kernel_invocation.cpp
  sycl/math.cpp
  sycl/sub_group.cpp
  sycl/sycl_test_suite.cpp 
  sycl/usm.cpp
  sycl/vec.cpp)

target_include_directories(sycl_tests PRIVATE ${Boost_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sycl_tests PRIVATE ${Boost_LIBRARIES})
add_sycl_to_target(TARGET sycl_tests)

add_executable(rt_tests 
  runtime/runtime_test_suite.cpp 
  runtime/dag_builder.cpp
  runtime/data.cpp)

target_include_directories(rt_tests PRIVATE ${Boost_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(rt_tests PRIVATE ${Boost_LIBRARIES})
add_sycl_to_target(TARGET rt_tests)
