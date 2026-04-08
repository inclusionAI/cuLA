// Copyright 2025-2026 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Inspired by
// https://github.com/Dao-AILab/flash-attention/blob/main/hopper/static_switch.h
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable
/// @param ...        - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```
#define BOOL_SWITCH(COND, CONST_NAME, ...)                                                         \
    [&] {                                                                                          \
        if (COND) {                                                                                \
            constexpr static bool CONST_NAME = true;                                               \
            return __VA_ARGS__();                                                                  \
        } else {                                                                                   \
            constexpr static bool CONST_NAME = false;                                              \
            return __VA_ARGS__();                                                                  \
        }                                                                                          \
    }()

/// @param COND           - a boolean expression (true → BF16Type, false → FP32Type)
/// @param BF16Type       - the type to use when COND is true
/// @param FP32Type       - the type to use when COND is false
/// @param TYPE_ALIAS     - a name given for the using type alias
/// @param ...            - code to execute
///
/// Usage:
/// ```
/// FP_TYPE_SWITCH(is_bf16, __nv_bfloat16, float, ElemType, [&] {
///     some_function<ElemType>(...);
/// });
/// ```
#define FP_TYPE_SWITCH(COND, BF16Type, FP32Type, TYPE_ALIAS, ...)                                  \
    [&] {                                                                                          \
        if (COND) {                                                                                \
            using TYPE_ALIAS = BF16Type;                                                           \
            return __VA_ARGS__();                                                                  \
        } else {                                                                                   \
            using TYPE_ALIAS = FP32Type;                                                           \
            return __VA_ARGS__();                                                                  \
        }                                                                                          \
    }()

/// Convenience shorthand: COND=true → __nv_bfloat16, COND=false → float
/// @param COND           - a boolean expression
/// @param TYPE_ALIAS     - a name given for the using type alias
/// @param ...            - code to execute
///
/// Usage:
/// ```
/// BETA_TYPE_SWITCH(params.is_beta_bf16, BetaType, [&] {
///     some_function<BetaType>(...);
/// });
/// ```
#define BETA_TYPE_SWITCH(COND, TYPE_ALIAS, ...)                                                    \
    FP_TYPE_SWITCH(COND, __nv_bfloat16, float, TYPE_ALIAS, __VA_ARGS__)
