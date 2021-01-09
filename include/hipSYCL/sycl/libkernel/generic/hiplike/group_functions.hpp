/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifdef SYCL_DEVICE_ONLY

#ifndef HIPSYCL_LIBKERNEL_DEVICE_GROUP_FUNCTIONS_HPP
#define HIPSYCL_LIBKERNEL_DEVICE_GROUP_FUNCTIONS_HPP

#include "../../backend.hpp"
#include "../../detail/data_layout.hpp"
#include "../../detail/thread_hierarchy.hpp"
#include "../../detail/warp_shuffle.hpp"
#include "../../id.hpp"
#include "../../sub_group.hpp"
#include <type_traits>

namespace hipsycl {
namespace sycl {

namespace detail {

// reduce implementation
template <typename Group, typename T, typename BinaryOperation,
          typename std::enable_if_t<!std::is_same_v<Group, sub_group>, int> = 0>
HIPSYCL_KERNEL_TARGET T group_reduce(Group g, T x, BinaryOperation binary_op,
                                     T *scratch) {

  auto lid = g.get_local_linear_id();
  size_t lrange = (g.get_local_range().size() + warpSize -1) / warpSize;
  sub_group sg{};

  x = group_reduce(sg, x, binary_op);
  if (sg.leader())
    scratch[lid / warpSize] = x;
  group_barrier(g);

  if (lrange != warpSize) {
    for (size_t i = lrange / 2; i > 0; i /= 2) {
      if (lid < i)
        scratch[lid] = binary_op(scratch[lid], scratch[lid + i]);
      group_barrier(g);
    }
  } else {
    if (lid < warpSize)
      x = group_reduce(sg, scratch[lid], binary_op);

    if (lid == 0)
      scratch[0] = x;

    group_barrier(g);
  }

  return scratch[0];
}

// any_of
template <typename Group, typename T>
HIPSYCL_KERNEL_TARGET bool any_of(Group g, T *first, T *last) {
  auto group_range = g.get_local_range().size();
  auto elements_per_group = (last - first + group_range - 1) / group_range;
  T *start_ptr = first + elements_per_group * g.get_local_linear_id();
  T *end_prt = start_ptr + elements_per_group;

  if (end_prt > last)
    end_prt = last;

  auto local = *start_ptr;

  for (T *p = start_ptr + 1; p < end_prt; ++p)
    local |= *p;

  return group_any_of(g, local);
}

template <typename Group, typename T, typename Predicate>
HIPSYCL_KERNEL_TARGET bool any_of(Group g, T *first, T *last, Predicate pred) {
  auto group_range = g.get_local_range().size();
  auto elements_per_group = (last - first + group_range - 1) / group_range;
  T *start_ptr = first + elements_per_group * g.get_local_linear_id();
  T *end_prt = start_ptr + elements_per_group;

  if (end_prt > last)
    end_prt = last;

  auto local = pred(*start_ptr);

  for (T *p = start_ptr + 1; p < end_prt; ++p)
    local |= pred(*p);

  return group_any_of(g, local);
}

// all_of
template <typename Group, typename T>
HIPSYCL_KERNEL_TARGET bool all_of(Group g, T *first, T *last) {
  auto group_range = g.get_local_range().size();
  auto elements_per_group = (last - first + group_range - 1) / group_range;
  T *start_ptr = first + elements_per_group * g.get_local_linear_id();
  T *end_prt = start_ptr + elements_per_group;

  if (end_prt > last)
    end_prt = last;

  auto local = *start_ptr;

  for (T *p = start_ptr + 1; p < end_prt; ++p)
    local &= *p;

  return group_all_of(g, local);
}

template <typename Group, typename T, typename Predicate>
HIPSYCL_KERNEL_TARGET bool all_of(Group g, T *first, T *last, Predicate pred) {
  auto group_range = g.get_local_range().size();
  auto elements_per_group = (last - first + group_range - 1) / group_range;
  T *start_ptr = first + elements_per_group * g.get_local_linear_id();
  T *end_prt = start_ptr + elements_per_group;

  if (end_prt > last)
    end_prt = last;

  auto local = pred(*start_ptr);

  for (T *p = start_ptr + 1; p < end_prt; ++p)
    local &= pred(*p);

  return group_all_of(g, local);
}

// none_of
template <typename Group, typename T>
HIPSYCL_KERNEL_TARGET bool none_of(Group g, T *first, T *last) {
  auto group_range = g.get_local_range().size();
  auto elements_per_group = (last - first + group_range - 1) / group_range;
  T *start_ptr = first + elements_per_group * g.get_local_linear_id();
  T *end_prt = start_ptr + elements_per_group;

  if (end_prt > last)
    end_prt = last;

  auto local = *start_ptr;

  for (T *p = start_ptr + 1; p < end_prt; ++p)
    local |= *p;

  return group_none_of(g, local);
}

template <typename Group, typename T, typename Predicate>
HIPSYCL_KERNEL_TARGET bool none_of(Group g, T *first, T *last, Predicate pred) {
  auto group_range = g.get_local_range().size();
  auto elements_per_group = (last - first + group_range - 1) / group_range;
  T *start_ptr = first + elements_per_group * g.get_local_linear_id();
  T *end_prt = start_ptr + elements_per_group;

  if (end_prt > last)
    end_prt = last;

  auto local = pred(*start_ptr);

  for (T *p = start_ptr + 1; p < end_prt; ++p)
    local |= pred(*p);

  return group_none_of(g, local);
}

// reduce
template <typename Group, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET T reduce(Group g, T *first, T *last,
                               BinaryOperation binary_op) {
  __shared__ char scratch_char[1024 / warpSize * sizeof(T)];
  T *scratch = reinterpret_cast<T *>(scratch_char);

  const size_t group_range = g.get_local_range().size();
  const size_t num_elements = last - first;
  const size_t elements_per_thread = (num_elements + group_range - 1) / group_range;
  const size_t lid = g.get_local_linear_id();

  T *start_ptr = first + lid;

  if (num_elements >= group_range) {
    auto local = *start_ptr;

    for (T *p = start_ptr + group_range; p < last; p+=group_range)
      local = binary_op(local, *p);

    return detail::group_reduce(g, local, binary_op, scratch);
  } else {
    const size_t warp_id = lid / warpSize;
    const size_t num_warps = group_range / warpSize;
    const size_t elements_per_warp = (num_elements + num_warps - 1) / num_warps;

    if (warp_id < num_warps) {
      const size_t active_threads = num_warps * warpSize;

      auto local = *start_ptr;

      for (T *p = start_ptr + active_threads; p < last; p+=active_threads)
        local = binary_op(local, *p);

      sub_group sg{};

      local = group_reduce(sg, local, binary_op);
      if (sg.leader())
        scratch[warp_id] = local;
      group_barrier(g);

      for (size_t i = num_warps / 2; i > 0; i /= 2) {
        if (lid < i)
          scratch[lid] = binary_op(scratch[lid], scratch[lid + i]);
        group_barrier(g);
      }
    }
    return scratch[0];
  }
}

template <typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET T reduce(Group g, V *first, V *last, T init,
                               BinaryOperation binary_op) {
  return binary_op(reduce(g, first, last, binary_op), init);
}

// exclusive_scan
template <typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET T *exclusive_scan(Group g, V *first, V *last, T *result,
                                        T init, BinaryOperation binary_op) {
  auto lid = g.get_local_linear_id();

  if (g.leader()) {
    *(result++) = init;
    while (first != last - 1) {
      *result = binary_op(*(result - 1), *(first++));
      result++;
    }
  }
  return group_broadcast(g, result);
}

template <typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET T *exclusive_scan(Group g, V *first, V *last, T *result,
                                        BinaryOperation binary_op) {
  return exclusive_scan(g, first, last, result, T{}, binary_op);
}

// inclusive_scan
template <typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET T *inclusive_scan(Group g, V *first, V *last, T *result,
                                        T init, BinaryOperation binary_op) {
  auto lid = g.get_local_linear_id();

  if (g.leader()) {
    if (first == last)
      return result;

    *(result++) = binary_op(init, *(first++));
    while (first != last) {
      *result = binary_op(*(result - 1), *(first++));
      result++;
      ;
    }
  }
  return group_broadcast(g, result);
}

template <typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET T *inclusive_scan(Group g, V *first, V *last, T *result,
                                        BinaryOperation binary_op) {
  return inclusive_scan(g, first, last, result, T{}, binary_op);
}

} // namespace detail

// broadcast
template <typename Group, typename T>
HIPSYCL_KERNEL_TARGET T group_broadcast(
    Group g, T x, typename Group::linear_id_type local_linear_id = 0) {
  __shared__ char scratch_char[sizeof(T)];
  T *scratch = reinterpret_cast<T *>(scratch_char);
  auto lid = g.get_local_linear_id();

  if (lid == local_linear_id)
    scratch[0] = x;
  group_barrier(g);

  return scratch[0];
}

template <typename Group, typename T>
HIPSYCL_KERNEL_TARGET T group_broadcast(Group g, T x,
                                        typename Group::id_type local_id) {
  auto target_lid = detail::linear_id<g.dimensions>::get(
      local_id, detail::get_local_size<g.dimensions>());

  return group_broadcast(g, x, target_lid);
}

// any_of
template <typename Group>
HIPSYCL_KERNEL_TARGET inline bool group_any_of(Group g, bool pred) {
  return __syncthreads_or(pred);
}

// all_of
template <typename Group>
HIPSYCL_KERNEL_TARGET inline bool group_all_of(Group g, bool pred) {
  return __syncthreads_and(pred);
}

// none_of
template <typename Group>
HIPSYCL_KERNEL_TARGET inline bool group_none_of(Group g, bool pred) {
  return !__syncthreads_or(pred);
}

// reduce
template <typename Group, typename T, typename BinaryOperation,
          typename std::enable_if_t<!std::is_same_v<Group, sub_group>, int> = 0>
HIPSYCL_KERNEL_TARGET T group_reduce(Group g, T x, BinaryOperation binary_op) {
  __shared__ char scratch_char[1024 / warpSize * sizeof(T)];
  T *scratch = reinterpret_cast<T *>(scratch_char);
  return detail::group_reduce(g, x, binary_op, scratch);
}

// exclusive_scan
template <typename Group, typename V, typename T, typename BinaryOperation,
          typename std::enable_if_t<!std::is_same_v<Group, sub_group>, int> = 0>
HIPSYCL_KERNEL_TARGET T group_exclusive_scan(Group g, V x, T init,
                                             BinaryOperation binary_op) {
  __shared__ char scratch_char[1024 / warpSize * sizeof(T)];
  T *scratch = reinterpret_cast<T *>(scratch_char);
  auto lid = g.get_local_linear_id();
  auto wid = lid / warpSize;
  size_t lrange = 1;
  auto group_local_range = g.get_local_range();
  for (int i = 0; i < g.dimensions; ++i)
    lrange *= group_local_range[i];

  sub_group sg{};

  auto local_x = group_inclusive_scan(sg, x, binary_op);
  auto last_wid = (wid + 1) * warpSize - 1;
  if (lid == (lrange < last_wid ? lrange - 1 : last_wid))
    scratch[wid] = local_x;
  group_barrier(g);

  if (lid < (lrange + warpSize - 1) / warpSize)
    scratch[lid] = group_inclusive_scan(sg, scratch[lid], binary_op);
  group_barrier(g);

  auto prefix = init;
  if (wid != 0)
    prefix = binary_op(init, scratch[wid - 1]);
  local_x = binary_op(prefix, local_x);

  local_x = detail::shuffle_up_impl(local_x, 1);
  if (lid % warpSize == 0)
    return prefix;
  return local_x;
}

// inclusive_scan
template <typename Group, typename T, typename BinaryOperation,
          typename std::enable_if_t<!std::is_same_v<Group, sub_group>, int> = 0>
HIPSYCL_KERNEL_TARGET T group_inclusive_scan(Group g, T x,
                                             BinaryOperation binary_op) {
  __shared__ char scratch_char[1024 / warpSize * sizeof(T)];
  T *scratch = reinterpret_cast<T *>(scratch_char);
  auto lid = g.get_local_linear_id();
  auto wid = lid / warpSize;
  size_t lrange = 1;
  auto group_local_range = g.get_local_range();
  for (int i = 0; i < g.dimensions; ++i)
    lrange *= group_local_range[i];

  sub_group sg{};

  auto local_x = group_inclusive_scan(sg, x, binary_op);
  auto last_wid = (wid + 1) * warpSize - 1;
  if (lid == (lrange < last_wid ? lrange - 1 : last_wid))
    scratch[wid] = local_x;
  group_barrier(g);

  if (lid < (lrange + warpSize - 1) / warpSize)
    scratch[lid] = group_inclusive_scan(sg, scratch[lid], binary_op);
  group_barrier(g);

  if (wid == 0)
    return local_x;
  return binary_op(scratch[wid - 1], local_x);
}

} // namespace sycl
} // namespace hipsycl

#endif // HIPSYCL_LIBKERNEL_DEVICE_GROUP_FUNCTIONS_HPP

#endif // SYCL_DEVICE_ONLY
