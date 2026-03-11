#pragma once

#include "tile_scheduler.h"

struct KDA_fwd_intra_params {
    using GmemShapeAkk  = cute::Shape<int32_t, int32_t, int32_t>;  // (seqlen_kv, seqlen_kv, h)
    using GmemStrideAkk = cute::Stride<int32_t, cute::_1, int32_t>;

    int total_q_len;
    int b;
    int h;
    int d;
    int chunk_size;
    float scale;

    void *__restrict__ q_ptr;             //[b, t, h, d]
    void *__restrict__ k_ptr;             //[b, t, h, d]
    void *__restrict__ g_ptr;             //[b, t, h, d]
    void *__restrict__ beta_ptr;          //[b, t, h]
    void *__restrict__ Aqk_out_ptr;       //[b, t, h, BT]
    void *__restrict__ Akk_out_ptr;       //[b, t, h, BT]
    void *__restrict__ cu_seqlens_ptr;    //[b + 1]
    void *__restrict__ chunk_indices_ptr; //[(b * t) / chunk_size, 2]

    GmemShapeAkk shape_Akk;
    GmemStrideAkk stride_Akk;

    StaticPersistentTileScheduler::Params tile_scheduler_params;

    int num_sm;
};