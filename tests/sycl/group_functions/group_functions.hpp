/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2020 Aksel Alpay and contributors
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef TESTS_GROUP_FUNCTIONS_HH
#define TESTS_GROUP_FUNCTIONS_HH

#include <cstddef>
#include <cstdio>
#include <functional>
#include <iostream>
#include <limits>
#include <math.h>
#include <type_traits>

#include <sstream>
#include <string>

using namespace cl;

#ifdef TESTS_GROUPFUNCTION_FULL
using test_types =
    boost::mpl::list<char, int, unsigned int, long long, float, double, sycl::vec<int, 1>,
                     sycl::vec<int, 2>, sycl::vec<int, 3>, sycl::vec<int, 4>, sycl::vec<int, 8>,
                     sycl::vec<short, 16>, sycl::vec<long, 3>, sycl::vec<unsigned int, 3>>;
#else
//using test_types = boost::mpl::list<char, unsigned int, float, double, sycl::vec<int, 2>>;
using test_types = boost::mpl::list<unsigned int>;
#endif

namespace detail {

template<typename T>
using elementType = std::remove_reference_t<decltype(T{}.s0())>;

template<typename T, int N>
std::string type_to_string(sycl::vec<T, N> v) {
  std::stringstream ss{};

  ss << "(";
  if constexpr (1 <= N)
    ss << +v.s0();
  if constexpr (2 <= N)
    ss << ", " << +v.s1();
  if constexpr (3 <= N)
    ss << ", " << +v.s2();
  if constexpr (4 <= N)
    ss << ", " << +v.s3();
  if constexpr (8 <= N) {
    ss << ", " << +v.s4();
    ss << ", " << +v.s5();
    ss << ", " << +v.s6();
    ss << ", " << +v.s7();
  }
  if constexpr (16 <= N) {
    ss << ", " << +v.s8();
    ss << ", " << +v.s9();
    ss << ", " << +v.sA();
    ss << ", " << +v.sB();
    ss << ", " << +v.sC();
    ss << ", " << +v.sD();
    ss << ", " << +v.sE();
    ss << ", " << +v.sF();
  }
  ss << ")";

  return ss.str();
}

template<typename T>
std::string type_to_string(T x) {
  std::stringstream ss{};
  ss << +x;

  return ss.str();
}

template<typename T, int N>
bool compare_type(sycl::vec<T, N> v1, sycl::vec<T, N> v2) {
  bool ret = true;
  if constexpr (1 <= N)
    ret &= v1.s0() == v2.s0();
  if constexpr (2 <= N)
    ret &= v1.s1() == v2.s1();
  if constexpr (3 <= N)
    ret &= v1.s2() == v2.s2();
  if constexpr (4 <= N)
    ret &= v1.s3() == v2.s3();
  if constexpr (8 <= N) {
    ret &= v1.s4() == v2.s4();
    ret &= v1.s5() == v2.s5();
    ret &= v1.s6() == v2.s6();
    ret &= v1.s7() == v2.s7();
  }
  if constexpr (16 <= N) {
    ret &= v1.s8() == v2.s8();
    ret &= v1.s9() == v2.s9();
    ret &= v1.sA() == v2.sA();
    ret &= v1.sB() == v2.sB();
    ret &= v1.sC() == v2.sC();
    ret &= v1.sD() == v2.sD();
    ret &= v1.sE() == v2.sE();
    ret &= v1.sF() == v2.sF();
  }

  return ret;
}

template<typename T>
bool compare_type(T x1, T x2) {
  return x1 == x2;
}

template<typename T, typename std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
HIPSYCL_KERNEL_TARGET
T initialize_type(T init) {
  return init;
}

template<typename T, typename std::enable_if_t<!std::is_arithmetic_v<T>, int> = 0>
HIPSYCL_KERNEL_TARGET
T initialize_type(elementType<T> init) {
  constexpr size_t N = T::get_count();

  if constexpr (std::is_same_v<elementType<T>, bool>)
    return T{init};

  if constexpr (N == 1) {
    return T{init};
  } else if constexpr (N == 2) {
    return T{init, init + 1};
  } else if constexpr (N == 3) {
    return T{init, init + 1, init + 2};
  } else if constexpr (N == 4) {
    return T{init, init + 1, init + 2, init + 3};
  } else if constexpr (N == 8) {
    return T{init, init + 1, init + 2, init + 3, init + 4, init + 5, init + 6, init + 7};
  } else if constexpr (N == 16) {
    return T{init,      init + 1,  init + 2,  init + 3, init + 4,  init + 5,
             init + 6,  init + 7,  init + 8,  init + 9, init + 10, init + 11,
             init + 12, init + 13, init + 14, init + 15};
  }

  static_assert(true, "invalide vector type!");
}

template<typename T, typename std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
HIPSYCL_KERNEL_TARGET
T get_offset(size_t margin, size_t divisor = 1) {
  if (std::numeric_limits<T>::max() <= margin) {
    return T{};
  }
  if constexpr (std::is_floating_point_v<T>) {
    return T{};
  }

  if constexpr (std::is_signed_v<T>) {
    return static_cast<T>(std::numeric_limits<T>::min() / divisor + margin);
  }
  return static_cast<T>(std::numeric_limits<T>::max() / divisor - margin);
}

template<typename T, typename std::enable_if_t<!std::is_arithmetic_v<T>, int> = 0>
HIPSYCL_KERNEL_TARGET
T get_offset(size_t margin, size_t divisor = 1) {
  using eT = elementType<T>;
  if (std::numeric_limits<eT>::max() <= margin) {
    return T{};
  }
  if constexpr (std::is_floating_point_v<eT>) {
    return T{};
  }

  if constexpr (std::is_signed_v<T>) {
    return initialize_type<T>(
        static_cast<eT>(std::numeric_limits<eT>::min() / divisor + margin));
  }
  return initialize_type<T>(
      static_cast<eT>(std::numeric_limits<eT>::max() / divisor - margin));
}

template<typename T>
HIPSYCL_KERNEL_TARGET
elementType<T> local_value(size_t id, size_t offsetBase) {
  const size_t N = T{}.get_count();

  auto offset = get_offset<elementType<T>>(offsetBase);
  return static_cast<elementType<T>>(id + offset);
}

inline void create_bool_test_data(std::vector<char> &buffer, size_t local_size,
                                  size_t global_size) {
  BOOST_REQUIRE(global_size == 4 * local_size);
  BOOST_REQUIRE(local_size + 10 < 2 * local_size);

  // create host_buf 4 different possible configurations:
  // 1: everything except one false
  // 2: everything false
  // 3: everything except one true
  // 4: everything true

  for (size_t i = 0; i < 2 * local_size; ++i)
    buffer[i] = false;
  for (size_t i = 2 * local_size; i < 4 * local_size; ++i)
    buffer[i] = true;

  buffer[10]                  = true;
  buffer[2 * local_size + 10] = false;

  BOOST_REQUIRE(buffer[0] == false);
  BOOST_REQUIRE(buffer[10] == true);
  BOOST_REQUIRE(buffer[local_size] == false);
  BOOST_REQUIRE(buffer[10 + local_size] == false);
  BOOST_REQUIRE(buffer[local_size * 2] == true);
  BOOST_REQUIRE(buffer[10 + local_size * 2] == false);
  BOOST_REQUIRE(buffer[local_size * 3] == true);
  BOOST_REQUIRE(buffer[10 + local_size * 3] == true);
}

template<typename T, int Line>
void check_binary_reduce(std::vector<T> buffer, size_t local_size, size_t global_size,
                         std::vector<bool> expected, std::string name,
                         size_t break_size = 0, size_t offset = 0) {
  std::vector<std::string> cases{"everything except one false", "everything false",
                                 "everything except one true", "everything true"};
  BOOST_REQUIRE(global_size / local_size == expected.size());
  for (size_t i = 0; i < global_size / local_size; ++i) {
    for (size_t j = 0; j < local_size; ++j) {
      // used to stop after first subgroup
      if (break_size != 0 && j == break_size)
        break;

      T computed      = buffer[i * local_size + j + offset];
      T expectedValue = initialize_type<T>(expected[i]);

      BOOST_TEST(compare_type(expectedValue, computed),
                 Line << ":" << type_to_string(computed) << " at position " << j
                      << " instead of " << type_to_string(expectedValue)
                      << " for case: " << cases[i] << " " << name);

      if (!compare_type(expectedValue, computed))
        break;
    }
  }
}

} // namespace detail

template<int N, int M, typename T>
class test_kernel;

template<int CallingLine, typename T, typename DataGenerator, typename TestedFunction,
         typename ValidationFunction>
void test_nd_group_function_1d(size_t local_size, size_t global_size, size_t offset_margin,
                               size_t offset_divisor, size_t buffer_size,
                               DataGenerator dg, TestedFunction f, ValidationFunction vf) {
  sycl::queue queue;
  std::vector<T> host_buf(buffer_size, T{});

  dg(host_buf);

  std::vector<T> original_host_buf(host_buf);

  {
    sycl::buffer<T, 1> buf{host_buf.data(), host_buf.size()};

    queue.submit([&](sycl::handler &cgh) {
      using namespace sycl::access;
      auto acc = buf.template get_access<mode::read_write>(cgh);

      cgh.parallel_for<class test_kernel<1, CallingLine, T>>(
        sycl::nd_range<1>{global_size, local_size},
        [=](sycl::nd_item<1> item) {
        auto g  = item.get_group();
        auto sg = item.get_sub_group();

        T local_value = acc[item.get_global_linear_id()];

        f(acc, item.get_global_linear_id(), sg, g, local_value);
      });
    });
  }

  vf(host_buf, original_host_buf);
}

template<int CallingLine, typename T, typename DataGenerator, typename TestedFunction,
         typename ValidationFunction>
void test_nd_group_function_2d(size_t local_size_x, size_t local_size_y,
                               size_t global_size_x, size_t global_size_y,
                               size_t offset_margin, size_t offset_divisor,
                               size_t buffer_size, DataGenerator dg, TestedFunction f,
                               ValidationFunction vf) {
  sycl::queue queue;
  std::vector<T> host_buf(buffer_size, T{});

  dg(host_buf);

  std::vector<T> original_host_buf(host_buf);

  {
    sycl::buffer<T, 1> buf{host_buf.data(), host_buf.size()};

    queue.submit([&](sycl::handler &cgh) {
      using namespace sycl::access;
      auto acc = buf.template get_access<mode::read_write>(cgh);

      cgh.parallel_for<class test_kernel<2, CallingLine, T>>(
        sycl::nd_range<2>{sycl::range<2>(global_size_x, global_size_y), sycl::range<2>(local_size_x, local_size_y)},
        [=](sycl::nd_item<2> item) {
        auto g                  = item.get_group();
        auto sg                 = item.get_sub_group();
        size_t custom_linear_id = item.get_local_linear_id() +
                                  local_size_x * local_size_y * item.get_group_linear_id();

        T local_value = acc[custom_linear_id];

        f(acc, custom_linear_id, sg, g, local_value);
      });
    });
  }

  vf(host_buf, original_host_buf);
}

#endif // TESTS_GROUP_FUNCTIONS_HH
