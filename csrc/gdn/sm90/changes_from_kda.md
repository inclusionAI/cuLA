## Important changes from KDA -> GDN

### Kept:

- The KK_inv lambda defined in mainloop is kept because it serves the same purpose of applying the beta. It is a modular function, so we don't need to worry about the scale applications - this needs to be changed in other lambda.

### Change:

- In load_kv, the cached state should be loaded NON-transposed. This is different from the kda implementation, c.f. mainloop_kda_fwd.hpp:909.
  - Thinking more on this, it's better to just align the state shape with how the KDA implementation has already carried it out - this means that the expected state shape should be made explicit higher up in the API chain, possibly in FLA.
- In Kimi linear, the K matrix is multipled with the cumulative alpha gating matrix everywhere - it is also per channel. In GDN the first I + (KK^T) matrix also involves multiplying by alpha, but it is on a per-sequence, per-head basis. Thus the K_scaled and Q_scaled need to be rewritten.
- Alpha (gate) has previous shape of (B, T, H, K) for Kimi Linear, with per-channel gating. GDN instead has per-head gating, so the shape becomes (B, T, H), and we instead load a vector of size blkSeqQ == blkSeqK into shared memory. This means the atoms and layouts related to Alpha must all be changed, as well as the application of the gate.
  - Because Alpha is now not the same shape as the Blk_Q/K/V tiles, it is now the same shape as the beta. This means that we don't need to create auxilliary layouts for the TMA loads, and instead we port over Alpha's SMEM layout into a CollectiveLoadVector
  - The load_qkv in mainloop_gdn_fwd.hpp doesn't load alpha anymore - this is transferred to the load_beta
  - extract_alpha_last needs to be changed to a simple index into the last index of shared alpha tensor, while checking for end of sequence boundaries. It just copies once.
  - Alpha params are now changed to pointers with gmemlayout instead of TMALoad type
  - Another alpha change - during GDN's forward pass, the gating matrix applied to KK^T is computed as the difference between  [i,j] coords in log space, then exp2f. However, KDA instead applies an elementwise mask that is pre computed. The final QK^T , also coputed in compute_aux_safe, doesn't multiply on the alpha gate matrix, so it is a normal tensor core multiplication instead.
  - SharedStorage needs to be changed in kernel
- Compute_aux_safe changes
  - In s2r_compute_subchunk_operandA, I tried to keep the changes as minimal as possible, so I kept the behavior of copying a tile of A, but I broadcast the alpha values, which are now a row equivalent, to all the 32 columns in the subchunk. This allows the previous broadcast_row_0 + exp2f(g - g_first) values to still work. POSSIBLE OPTIMIZATION: It might be faster to just use the same register + identity tensor + row indexing across threads that
- Compute_loop_body changes
  - The alpha loading in + scaling needs to be changed, since the KDA implementation loads in a tile of 32 across the head dimension. I did the same change that i did in compute_aux_safe to create a dummy tensor shape that broadcasts.
  - I also stopped using a CopyAlpaAhtom and instead do a manual unrolled loop when loading in the shared alpha values for QK scaling
- KV state shape:
  - It looks like KDA implementation uses the same V^T * K_scaled, with the KV_state shape being d_V x d_K in the output. This is also equivalent to the FLA transpose flag being set to TRUE.
- Change in kernel_gdn_fwd.hpp:
  - Because the load type for alpha is now a predicated vector, we need to also change the alphapipelineparam initializaiton, moved it down next to beta, since they are loaded together

### Possible Optimizations:

- Because GDN doesn't need to materialize an entire register tile to hold results, we can load in the rows directly from shared memory and not worry about copying through to registers before multiplying. This could allow more aggressive use of the register file, in exchange for added latency from accessing SMEM. To keep consistency with the previous KDA implementation, I just used 0-strides to broadcast along the row dimension.

