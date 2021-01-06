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
#ifdef HIPSYCL_PLATFORM_HIP

#ifndef HIPSYCL_LIBKERNEL_HIP_GROUP_FUNCTIONS_HPP
#define HIPSYCL_LIBKERNEL_HIP_GROUP_FUNCTIONS_HPP

#include "../backend.hpp"
#include "../id.hpp"
#include "../sub_group.hpp"
#include "../vec.hpp"
#include <type_traits>

namespace hipsycl {
namespace sycl {

namespace detail {

template <typename T, typename operation>
__device__ T apply_on_data(T x, operation op) {
  constexpr int words_no = (sizeof(T) + sizeof(int) - 1) / sizeof(int);

  int words[words_no];
  __builtin_memcpy(words, &x, sizeof(T));

#pragma unroll
  for (int i = 0; i < words_no; i++)
    words[i] = op(words[i]);

  T output;
  __builtin_memcpy(&output, words, sizeof(T));

  return output;
}

// implemented based on warp_shuffle_op in rocPRIM
template <typename T> __device__ T shuffle_impl(T x, int id) {
  return apply_on_data(x, [id](int data) { return __shfl(data, id); });
}

// dpp sharing instruction abstraction based on rocPRIM
// the dpp_ctrl can be found in the GCN3 ISA manual
template <typename T, int dpp_ctrl, int row_mask = 0xf, int bank_mask = 0xf,
          bool bound_ctrl = false>
__device__ T warp_move_dpp(T x) {
  return apply_on_data(x, [=](int data) {
    return __builtin_amdgcn_update_dpp(0, data, dpp_ctrl, row_mask, bank_mask,
                                       bound_ctrl);
  });
}

} // namespace detail

// broadcast
template <typename T>
HIPSYCL_KERNEL_TARGET T group_broadcast(
    sub_group g, T x, typename sub_group::linear_id_type local_linear_id = 0) {
  return detail::shuffle_impl(x, static_cast<int>(local_linear_id));
}

template <typename T, int N>
HIPSYCL_KERNEL_TARGET sycl::vec<T, N>
group_broadcast(sub_group g, sycl::vec<T, N> x,
                typename sub_group::linear_id_type local_linear_id = 0) {
  return detail::shuffle_impl<T, N>(x, static_cast<int>(local_linear_id));
}

// barrier
template <typename Group>
HIPSYCL_KERNEL_TARGET inline void
group_barrier(Group g, memory_scope fence_scope = Group::fence_scope) {
  if (fence_scope == memory_scope::device) {
    __threadfence_system();
  }
  __syncthreads();
}

template <>
HIPSYCL_KERNEL_TARGET inline void group_barrier(sub_group g,
                                                memory_scope fence_scope) {
  if (fence_scope == memory_scope::device) {
    __threadfence_system();
  } else if (fence_scope == memory_scope::work_group) {
    __threadfence_block();
  }
  // threads run in lock-step no sync needed
}

// any_of
template <>
HIPSYCL_KERNEL_TARGET inline bool group_any_of(sub_group g, bool pred) {
  return __any(pred);
}

// all_of
template <>
HIPSYCL_KERNEL_TARGET inline bool group_all_of(sub_group g, bool pred) {
  return __all(pred);
}

// none_of
template <>
HIPSYCL_KERNEL_TARGET inline bool group_none_of(sub_group g, bool pred) {
  return !__any(pred);
}

// reduce
template <typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET T group_reduce(sub_group g, T x,
                                     BinaryOperation binary_op) {
  auto local_x = x;

  if (__ballot(1) == 0xFFFFFFFFFFFFFFFF) {
    // adaption of rocprim dpp_reduce
    // quad_perm: add 0+1, 2+3
    local_x = binary_op(detail::warp_move_dpp<T, 0xb1>(local_x), local_x);
    // quad_perm: add 0+2
    local_x = binary_op(detail::warp_move_dpp<T, 0x4e>(local_x), local_x);
    // row_sr: add 0+4
    local_x = binary_op(detail::warp_move_dpp<T, 0x114>(local_x), local_x);
    // row_sr: add 0+8
    local_x = binary_op(detail::warp_move_dpp<T, 0x118>(local_x), local_x);
    // row_bcast15: add 0+15
    local_x = binary_op(detail::warp_move_dpp<T, 0x142>(local_x), local_x);

    if (warpSize > 32) {
      // row_bcast31: add 0+31
      local_x = binary_op(detail::warp_move_dpp<T, 0x143>(local_x), local_x);
    }

    // get the result from last thead
    return detail::shuffle_impl(local_x, 63);
  } else {
    auto lid = g.get_local_linear_id();

    size_t lrange = 1;
    auto group_local_range = g.get_local_range();
    for (int i = 0; i < g.dimensions; ++i)
      lrange *= group_local_range[i];

    group_barrier(g);

    for (size_t i = lrange / 2; i > 0; i /= 2) {
      auto other_x = detail::shuffle_impl(local_x, lid + i);
      if (lid < i)
        local_x = binary_op(local_x, other_x);
    }
    return detail::shuffle_impl(local_x, 0);
  }
}

// exclusive_scan
template <typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET T group_exclusive_scan(sub_group g, V x, T init,
                                             BinaryOperation binary_op) {
  auto lid = g.get_local_linear_id();
  size_t lrange = g.get_local_linear_range();

  auto local_x = x;

  for (size_t i = 1; i < lrange; i *= 2) {
    size_t next_id = lid - i;
    if (i > lid)
      next_id = 0;

    auto other_x = detail::shuffle_impl(local_x, next_id);
    if (i <= lid && lid < lrange)
      local_x = binary_op(local_x, other_x);
  }

  size_t next_id = lid - 1;
  if (g.leader())
    next_id = 0;

  auto return_value = detail::shuffle_impl(local_x, lid - 1);

  if (g.leader())
    return init;

  return binary_op(return_value, init);
}

// inclusive_scan
template <typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET T group_inclusive_scan(sub_group g, T x,
                                             BinaryOperation binary_op) {
  auto local_x = x;
  auto lid = g.get_local_linear_id();
  auto row_id = lid % warpSize;

  if (__ballot(1) == 0xFFFFFFFFFFFFFFFF) {
    // adaption of rocprim dpp_scan
    T tmp;
    // row_sr:1
    tmp = binary_op(detail::warp_move_dpp<T, 0x111>(local_x), local_x);
    if (row_id > 0)
      local_x = tmp;

    // row_sr:2
    tmp = binary_op(detail::warp_move_dpp<T, 0x112>(local_x), local_x);
    if (row_id > 1)
      local_x = tmp;

    // row_sr:4
    tmp = binary_op(detail::warp_move_dpp<T, 0x114>(local_x), local_x);
    if (row_id > 3)
      local_x = tmp;

    // row_sr:8
    tmp = binary_op(detail::warp_move_dpp<T, 0x118>(local_x), local_x);
    if (row_id > 7)
      local_x = tmp;

    // row_bcast15
    tmp = binary_op(detail::warp_move_dpp<T, 0x142>(local_x), local_x);
    if (row_id % 32 >= 16)
      local_x = tmp;

    if (warpSize > 32) {
      // row_bcast31
      tmp = binary_op(detail::warp_move_dpp<T, 0x143>(local_x), local_x);
      if (row_id >= 32)
        local_x = tmp;
    }

    return local_x;
  } else {
    size_t lrange = g.get_local_linear_range();

    auto local_x = x;

    for (size_t i = 1; i < lrange; i *= 2) {
      size_t next_id = lid - i;
      if (i > lid)
        next_id = 0;

      auto other_x = detail::shuffle_impl(local_x, next_id);
      if (i <= lid && lid < lrange)
        local_x = binary_op(local_x, other_x);
    }

    return local_x;
  }
}

} // namespace sycl
} // namespace hipsycl

#endif // HIPSYCL_LIBKERNEL_HIP_GROUP_FUNCTIONS_HPP

#endif // HIPSYCL_PLATFORM_HIP
#endif // SYCL_DEVICE_ONLY
