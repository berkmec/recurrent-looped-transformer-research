# Core mathematical contract (before implementation)

At token t, encoder E maps the observed prefix x_1:t to e_t with causal self-attention. Encoder layer KV and projected global memory M_t are append-only during a forward pass. Decoder state H_t=(s_t,C^D_t); each layer cache retains W-1 preceding input keys/values, W includes the current position.

r_t=RMSNorm(s_(t-1)); g_t=sigmoid(G[e_t;r_t]); z_t=e_t+alpha*g_t*A*r_t, alpha=.1. Decoder applies residual SWA, causal global cross-attention to M_t, then residual GELU FFN. s_t is decoder output and RMSNorm/readout yields prefix state-class logits. The no-hidden-feedback control sets the entire additive merge to zero but keeps the same encoder, global memory, decoder SWA and allocated parameters. Its inactive merge and initial-state parameters are reported separately.

Full gradients traverse recurrent s, decoder cache, encoder attention and global memory. TBPTT with horizon 8 detaches s and decoder cache at boundaries; it intentionally retains the encoder/global-memory paths, and therefore is not a claim that every long-distance gradient has been removed. Streaming inference never detaches to approximate a model. Full-prefix encoder and incremental encoder should agree in outputs and parameter gradients.

Parity oracle repeatedly XORs the state with binary operations; symbol 2 is neutral. Five-state oracle composes two explicit noncommuting permutations from initial state 0: cycle (0 1 2 3 4), transposition (0 1), and identity symbol 2. Targets are independently generated sequential states; no oracle is in learned inference. Training uses all prefix targets and final accuracy remains primary. One operation is one consumed token; added neutral tokens increase t but not useful operation count d. No k-like title refinement exists in this core; k=0 throughout.

Independent deviations: one untied encoder/decoder layer, width32 or48, heads2, GELU FFN twice width, no dropout, small classification output, no tokenizer, no large-model reproduction. RoPE is used on self-attention and cross-memory query/keys.
