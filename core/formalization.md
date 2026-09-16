# Core mathematical contract (before implementation)

At token t, encoder E maps the observed prefix x_1:t to e_t with causal self-attention. Encoder layer KV and projected global memory M_t are append-only during a forward pass. Decoder state H_t=(s_t,C^D_t); each layer cache retains W-1 preceding input keys/values, W includes the current position.

r_t=RMSNorm(s_(t-1)); g_t=sigmoid(G[e_t;r_t]); z_t=e_t+alpha*g_t*A*r_t, alpha=.1. Decoder applies residual SWA, causal global cross-attention to M_t, then residual GELU FFN. s_t is decoder output and RMSNorm/readout yields prefix state-class logits. The no-hidden-feedback control sets the entire additive merge to zero but keeps the same encoder, global memory, decoder SWA and allocated parameters. Its inactive merge and initial-state parameters are reported separately.

Full gradients traverse recurrent s, decoder cache, encoder attention and global memory. TBPTT with horizon 8 detaches s and decoder cache at boundaries; it intentionally retains the encoder/global-memory paths, and therefore is not a claim that every long-distance gradient has been removed. Streaming inference never detaches to approximate a model. Full-prefix encoder and incremental encoder should agree in outputs and parameter gradients.

Parity oracle repeatedly XORs the state with binary operations; symbol 2 is neutral. Five-state oracle composes two explicit noncommuting permutations from initial state 0: cycle (0 1 2 3 4), transposition (0 1), and identity symbol 2. Targets are independently generated sequential states; no oracle is in learned inference. Training uses all prefix targets and final accuracy remains primary. One operation is one consumed token; added neutral tokens increase t but not useful operation count d. No k-like title refinement exists in this core; k=0 throughout.

Independent deviations: one untied encoder/decoder layer, width32 or48, heads2, GELU FFN twice width, no dropout, small classification output, no tokenizer, no large-model reproduction. RoPE is used on self-attention and cross-memory query/keys.

## Gradient-route clarification (append-only explanatory addendum)

For the augmented decoder state, the recurrent-only derivative contains the ordered product J_T J_(T-1)...J_(i+1), where J_t = partial H_t / partial H_(t-1), holding encoder memory fixed. The total input derivative also includes encoder self-attention and global-memory paths; it is not equal to that product alone. Boundary detachment replaces the relevant s/cache derivative by zero, while partial L / partial M_i can remain nonzero. This is why truncating decoder recurrence does not imply every early-input gradient vanishes.

Dense prefix loss is (1/T) sum_t CE(readout(s_t), y_t); delayed loss is CE(readout(s_T), y_T). Dense loss supplies many shorter credit-assignment paths and32times as many supervised targets at T=32. The separately frozen delayed-loss companion compares full and truncated gradients within that regime; it does not hold target count constant relative to dense supervision. No claim follows that nonzero, small or large gradient norms alone demonstrate learning or reasoning depth.

The step object transports encoder/global caches alongside H for convenience. All training/evaluation forwards rebuild caches from fresh state under current weights. This small step API has no parameter-version stamp: external callers must not reuse state after updating weights. Current-policy RL replay and a production serving cache manager are outside this experiment.

## Workload and depth terminology clarification (reporting addendum)

The earlier phrase "useful operation count d" denotes only a workload count in this core experiment, not graph dependency depth or a proved lower bound on the shortest algorithm. Report actual consumed token count t, designated non-neutral input count, and actual oracle state-change count separately. Parity bit0 is an identity; the five-state transposition also fixes states2,3,4. Even the number of actual state changes is not a minimal sequential-depth theorem. Neutral suffixes leave the original task and its answer fixed, increase t, and keep k=0. Graph/title dependency depth used by the separate reasoning-window experiments is a distinct structural quantity and cannot be substituted for any of these counts.
