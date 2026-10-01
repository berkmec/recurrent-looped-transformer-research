# U4-inject denetimi — 11 PASS / 0 FAIL

- **1-freeze** — PASS: 8 sources, hash 523dba46f4486746; problems=none
- **2-completeness** — PASS: 9/9 conditions evaluated
- **3-integrity** — PASS: 9 conditions: checkpoint/updates hashes and 2048 rows each; problems=none
- **4-headline** — PASS: arms that learned (validation > 0.9): additive 3/3, inject 0/3, inject_mix 0/3
- **5-chance** — PASS: all 6 injection validations within 0.12 of the 0.200 chance level: [0.234, 0.277, 0.246, 0.215, 0.309, 0.215]
- **6-no-residual-explosion** — PASS: control state RMS growth across a 512-step neutral run: [0.97, 1.2, 1.03] - no growth in 3/3, so the premise the cycle was built on is absent from the baseline
- **7-injection-shrinks** — PASS: injection arms shrink instead: [0.31, 0.48, 0.47, 0.4, 0.27, 0.27]
- **8-gradient-confound** — PASS: over the first 64 steps, before either arm learned: loss 1.6066 vs 1.6085, median gradient 0.145 vs 0.146 - indistinguishable, so the later gradient difference cannot be attributed to the mechanism
- **9-peak-vs-median** — PASS: the control's LATE median gradient 0.0018 is below the inject arm's 0.2130; its 1.78e4 figure is a rare spike, not its typical value
- **10-state-is-query-only** — PASS: source confirms both injection arms build `a` from a query on the previous state over the INPUT-derived k/v, and that inject_mix mixes e with a - so neither arm passes the state's content, and neither tests channel width
- **11-two-terms-not-one** — PASS: measured by running the decoders, not by matching source text: with one input and two different previous states, the sealed Core window cache CHANGES (True) because its k/v come from the post-feedback z, while inject (False) and inject_mix (False) leave it bitwise identical because theirs come from e. So the arms differ in two terms - how the state enters AND whether it reaches the sliding-window cache - and the 6/6 failure cannot be attributed to the injection mechanism alone. The arms do remain recurrent through the query (output changes: {'inject': True, 'inject_mix': True}), so this is a confound, not an inert arm. Report section 7 states the bound; the separating arm (injection plus k/v from the post-injection z) is not built.
