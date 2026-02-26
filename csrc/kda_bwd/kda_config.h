#pragma once

#include "tile_scheduler.h"

struct KDA_bwd_intra_params {
    using index_t = int64_t;

    int total_q_len;
    int b;
    int h;
    int d, d_v; 
    int chunk_size;

    void *__restrict__ q_ptr;             //[b, t, h, d]
    void *__restrict__ k_ptr;             //[b, t, h, d]
    void *__restrict__ g_ptr;             //[b, t, h, d]
    void *__restrict__ beta_ptr;          //[b, t, h]
    void *__restrict__ dAqk_ptr;          //[b, t, h, BT]
    void *__restrict__ dAkk_ptr;          //[b, t, h, BT]
    void *__restrict__ dq_ptr;            //[b, t, h, d]
    void *__restrict__ dk_ptr;            //[b, t, h, d]
    void *__restrict__ db_ptr;            //[b, t, h]
    void *__restrict__ dg_ptr;            //[b, t, h, d]
    void *__restrict__ dq_out_ptr;        //[b, t, h, d]
    void *__restrict__ dk_out_ptr;        //[b, t, h, d]
    void *__restrict__ db_out_ptr;        //[b, t, h]
    void *__restrict__ dg_out_ptr;        //[b, t, h, d]
    void *__restrict__ cu_seqlens_ptr;    //[b + 1]
    void *__restrict__ chunk_indices_ptr; //[len(cu_seqlens) - 1]

    NaiveTileScheduler::Params tile_scheduler_params;

    int num_sm;
};

// Parameters for KDA backward WY dqkg fused kernel
// This kernel fuses the WY backward computation with dq/dk/dg gradient computation
struct KDA_bwd_wy_dqkg_fused_params {
    using index_t = int64_t;

    int total_q_len;
    int b;
    int h;
    int d;     // K dimension
    int d_v;   // V dimension
    int chunk_size;

    float scale;

    // Input tensors
    void *__restrict__ q_ptr;             //[b, t, h, d] bf16
    void *__restrict__ k_ptr;             //[b, t, h, d] bf16
    void *__restrict__ v_ptr;             //[b, t, h, d_v] bf16
    void *__restrict__ v_new_ptr;         //[b, t, h, d_v] bf16
    void *__restrict__ g_ptr;             //[b, t, h, d] fp32
    void *__restrict__ beta_ptr;          //[b, t, h] bf16
    void *__restrict__ A_ptr;             //[b, t, h, BT] bf16 (Akk matrix)
    void *__restrict__ h_ptr;             //[NT, h, d, d_v] fp32 - chunk-level hidden states
    void *__restrict__ do_ptr;            //[b, t, h, d_v] bf16
    void *__restrict__ dh_ptr;            //[NT, h, d, d_v] fp32 - chunk-level hidden gradients
    void *__restrict__ dv_ptr;            //[b, t, h, d_v] bf16 - gradients from previous kernel

    // Output tensors
    void *__restrict__ dq_out_ptr;        //[b, t, h, d] fp32
    void *__restrict__ dk_out_ptr;        //[b, t, h, d] fp32
    void *__restrict__ dv_out_ptr;        //[b, t, h, d_v] bf16
    void *__restrict__ db_out_ptr;        //[b, t, h] fp32
    void *__restrict__ dg_out_ptr;        //[b, t, h, d] fp32
    void *__restrict__ dA_out_ptr;        //[b, t, h, BT] fp32

    // Sequence/chunk info
    void *__restrict__ cu_seqlens_ptr;    //[b + 1]
    void *__restrict__ chunk_indices_ptr; //[NT * 2]

    NaiveTileScheduler::Params tile_scheduler_params;

    int num_sm;
};