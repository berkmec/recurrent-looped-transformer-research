# Core research execution — IN PROGRESS

Primary protocol SHA256: 51b39ee9debb8769148c0d18d92f2adf91be8ef5e052ee58fd9d93a04c808a3f.

Packages1–4 and the core-width portion of14. Independent implementation follows the pinned source's state/cache contract; this is not official result reproduction. State is (last decoder hidden, layerwise SWA), and encoder/global memory remains causal. The nofeedback control retains encoder, SWA and cross-attention, bypassing all hidden merge modules. Its allocated but inactive parameters are counted separately.

10 contract/diagnostic tests passed; test-first RED and GREEN logs preserved. Real training started CPU1thread, 21 primary runs, each2000 optimizer updates, batch32, token length32, seeds17/43/91. Primary tasks parity and explicit noncommuting five-state permutations have one neutral identity symbol sampled with probability.125. All prefixes supervised. Width32, second width48 on parity, 1encoder+1decoder, heads2, FF2width, SWA4, feedback alpha.1. Selected checkpoint minimizes fixed validation prefix CE, without consulting test results. Prefix targets make primary TBPTT a local-transition supervised comparison; the separate final-only companion tests delayed-error learning.

Before final evaluation, package2 remains UNMEASURED. Length32 validation learned in initial seed17 RLT run; this is validation evidence only. Planned held-out lengths32/64/128/256/512/1024 with512 examples per seed/task/condition. No infinite-depth claim. Primary protocol frozen before held-out generation.

Package3 appends neutral identity tokens to the exact same base problem: t grows, useful nonidentity operation count remains unchanged, k=0. This is token-clock computation, not title refinement. Output is one state-class readout, not generated natural-language output. Time is measured under concurrent CPU load unless an isolated repeat is explicitly recorded.

Package4 includes full versus s/SWA-truncated BPTT8 with all3seeds, and a separate final-target-only 2000step companion full vsTBPTT8. TBPTT retains gradients through encoder/global memory. Diagnostic lag gradients and finite differences use oracle final labels; nonzero gradients do not demonstrate learned long-distance reasoning. Whole-forward activation recomputation has output/gradient equivalence tests; CPU peak activation memory is UNMEASURED.

Checkpoint files, hashes, event wall times, predictions/targets and per-seed metrics are being written below core/. No test accuracy has been inferred from validation. No energy, large-scale language-model, RL, title refinement or unrestricted natural-language claim.
