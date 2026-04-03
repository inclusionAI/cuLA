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

#pragma once

#include "kda/sm100/tile_scheduler.hpp"

struct KDA_fwd_intra_params {
    using GmemShapeAkk = cute::Shape<int32_t, int32_t, int32_t>;  // (seqlen_kv, seqlen_kv, h)
    using GmemStrideAkk = cute::Stride<int32_t, cute::_1, int32_t>;

    int total_q_len;
    int b;
    int h;
    int d;
    int chunk_size;
    float scale;
    bool use_tf32_inverse;
    bool unified_gref;

    void* __restrict__ q_ptr;              //[b, t, h, d]
    void* __restrict__ k_ptr;              //[b, t, h, d]
    void* __restrict__ g_ptr;              //[b, t, h, d]
    void* __restrict__ beta_ptr;           //[b, t, h]
    void* __restrict__ Aqk_out_ptr;        //[b, t, h, BT]
    void* __restrict__ Akk_out_ptr;        //[b, t, h, BT]
    void* __restrict__ cu_seqlens_ptr;     //[b + 1]
    void* __restrict__ chunk_indices_ptr;  //[(b * t) / chunk_size, 2]

    GmemShapeAkk shape_Akk;
    GmemStrideAkk stride_Akk;

    StaticPersistentTileScheduler::Params tile_scheduler_params;

    int num_sm;
};

struct KDA_fwd_recomp_w_u_params {
    using GmemShapeWUKg = cute::Shape<int32_t, int32_t, int32_t>;  // (seqlen_kv, seqlen_kv, h)
    using GmemStrideWUKg = cute::Stride<int32_t, cute::_1, int32_t>;

    int total_len;
    int b;
    int h;
    int d;
    int chunk_size;

    void* __restrict__ k_ptr;              //[b, t, h, d]
    void* __restrict__ v_ptr;              //[b, t, h, d]
    void* __restrict__ q_ptr;              //[b, t, h, d]  (optional, for StoreQG)
    void* __restrict__ beta_ptr;           //[b, t, h]
    void* __restrict__ A_ptr;              //[b. t, h, BT]
    void* __restrict__ g_ptr;              //[b, t, h, d]
    void* __restrict__ cu_seqlens_ptr;     //[b + 1]
    void* __restrict__ chunk_indices_ptr;  //[(b * t) / chunk_size, 2]
    void* __restrict__ w_out_ptr;          //[b, t, h, d]
    void* __restrict__ u_out_ptr;          //[b, t, h, d]
    void* __restrict__ kg_out_ptr;         //[b, t, h, d]
    void* __restrict__ qg_out_ptr;         //[b, t, h, d]  (optional, for StoreQG)

    bool store_qg;

    GmemShapeWUKg shape_wukg;
    GmemStrideWUKg stride_wukg;

    StaticPersistentTileScheduler::Params tile_scheduler_params;

    int num_sm;
};