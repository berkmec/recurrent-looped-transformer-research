# R6 independent memory/gradient review — preliminary pre-freeze

Date: 2026-09-17. Scope: task-3-brief.md, task-3-context.md, memory-review-brief.md; review_cycle/memory_bench.py, review_cycle/test_memory_bench.py, immutable core/model.py, existing task-3-calibration evidence. CPU-only bounded review, one torch thread, no GPU access and no source edits. notes/R6-report.md did not yet exist when inspected.

## Verdict

PASS for the inspected gradient-route implementation and bounded CPU checks. No source-level gradient-equivalence blocker found. This is NOT approval of GPU equivalence, final frozen main measurements, or GPU efficiency claims. Final hash-specific review and observed GPU pilot/validation evidence are still required before main. The already observed CPU pilots use an older runner hash and cannot certify the currently inspected source.

Reviewed SHA256:
- review_cycle/memory_bench.py: 3e5569e94e32f21800ddd85cc09867a7ae1ed5968640943eb092e3d4118bc680
- review_cycle/test_memory_bench.py: 381a8418c2b58c8016fa51cba26c0daa87fde6697a2654c0fb903bf7bbe75d76
- core/model.py: 121f11fcb3c1308b7f67d898f2aa09c37226d3b0db27dd4e5bf1ebeaea281f43

## Concrete findings and interpretation limits

1. OOM interrupted-phase telemetry is incomplete (memory_bench.py:316-318). The handler records process RAM and completed phase rows, but does not sample memory_allocated/reserved/max_memory_allocated/max_memory_reserved for the phase that actually failed. A forward OOM may therefore retain only the ready row. This does not invalidate the explicit OOM status or justify reducing batch; it means no measured CUDA peak can be claimed for the failed condition. If failed-run peak accounting is part of the final deliverable, capture current CUDA counters defensively in the handler before freeze. Otherwise explicitly label OOM conditions as categorical capacity boundaries with unknown completed-update peak. Nonblocking only with that reporting restriction.

2. Segment latency includes an implicit device synchronization per chunk (memory_bench.py:143, float(value.detach())). This overhead scales with length/chunk, whereas full/detach/checkpoint have one scalar conversion per whole loss. The measured latency is still actual latency of this implementation, but cannot establish an intrinsic segment/checkpoint algorithm speed comparison without disclosing that extra synchronization. If changing this implementation detail, do so before freeze and repeat final source-specific validation; no post-freeze retuning.

3. host_cpu_observation is now implemented (218-228, 280, 321-325), so the prior missing-observation concern is resolved in the inspected source. It is system-wide CPU busy fraction over the whole worker interval including warmup/checkpoint I/O, not per measured update, GPU utilization, competing-process identification, or proof of GPU exclusivity. Keep the parent’s external GPU ownership record and state that scope. Generic FAIL exits at 319-320 do not include this ending observation; their stderr and error result remain available.

4. Existing calibration is stale for final-source authority. All seven saved CPU pilot conditions report PASS, five optimizer updates, initialized optimizer before three measured updates, but runner SHA256 is 44455db89fca9deb6a5c2eef6c0645c6dfc720c7d67a6e6fb8b8e55dbf00d65f and host_cpu_observation is absent. They are useful development evidence only. Current source is pinned above; final freeze/validation/pilot should reference one consistent source set.

## Gradient route review

- Full and whole-forward nonreentrant checkpoint call the same Core forward with full recurrence.
- Segment global_inputs computes e/mk/mv once; detached leaves preserve decoder access. Each chunk CE uses sum divided by the same total y.numel(), including a shorter last chunk (116-151).
- Only s/dk/dv are detached across boundaries, matching immutable Core.forward. Dropping state dictionary mk/mv/t is harmless here: decode receives global prefixes and t explicitly and consumes state only for s/dk/dv.
- Decoder gradients accumulate directly across chunks. Leaf gradients accumulate over all decoder uses. A single multi-output autograd.backward(original, leaf gradients) sends both direct e contribution and mk/mv projection contributions back through the original shared encoder graph once, without duplicating the indirect paths.
- Encoder checkpoint wraps precisely global_inputs. Its recomputation preserves the same route; no claim that whole checkpoint and truncation are equivalent.
- Unit tests compare every named parameter gradient, every parameter after AdamW updates, and step/exp_avg/exp_avg_sq across three updates in float64 at nonmultiple length 19. Negative controls detect full versus truncated route difference and removal of the global bridge.

## Measurement review

- Main launch creates a fresh subprocess per condition, fixed batch/precision/length, independently seeded paired update data, two actual optimizer warmups and three actual measured optimizer updates.
- AdamW state initialization is verified before measured work; initial/final states and final gradients are saved. Equivalence compares final model, optimizer, gradients and measured losses for all three intended pairings, at lengths 32/128. Main requires matching equivalence PASS, protocol hash and source hashes.
- CUDA synchronization and reset of phase peaks are explicit. Max across reset phase peaks reconstructs whole measured-update peak. Allocated, reserved and process working set/commit remain separate. Process lifetime peak is labeled as lifetime, not reset phase peak. Segment forward/backward is an intentionally merged phase.
- Primary runs have no saved-tensor hooks. Diagnostic runs have separate condition names/processes. SavedStorageTracker deduplicates aliases by underlying storage identity, refcounts save lifetime, and detaches packed tensors to avoid graph cycles. Its peak is saved backing-storage lifetime, not cumulative saves and not total resident VRAM. Parameter/input saves are included; unsaved allocations excluded. Checkpoint-internal hook shadowing is explicitly disclosed.
- Source/protocol hashes and snapshots, PID, stdout/stderr, update receipts and no-retry OOM accounting are present. No hidden batch reduction was found.

## Checks actually executed

1. C:/Users/pc/Documents/ChatGPT/DaxDream/.venv/Scripts/python.exe -B -m unittest review_cycle.test_memory_bench -v
   Result: 7/7 PASS, reported test time 5.159 s. CPU only, one torch thread. Includes actual CPU worker update, route/optimizer checks, storage alias/lifetime checks and protocol tamper detection.

2. Read-only comparison of existing length32 CPU pilot final.pt payloads, including 175 tensor entries across model/optimizer/gradients, rtol=3e-4 and atol=3e-6:
   - full versus checkpoint: PASS; maximum absolute difference 0.
   - detach versus segment: PASS; maximum absolute difference 2.6263296604156494e-07.
   - detach versus segment_encoder_checkpoint: PASS; same maximum absolute difference.
   These results belong to the older calibration source hash identified above.

3. Existing encoder-checkpoint diagnostic pilot has live saved storage 0 after measured updates; first reported peak_unique_storage_bytes 413284, save_requests 2498. These are CPU saved-storage diagnostics, not GPU memory measurements.

## Required final re-review inputs

- Final source hashes and frozen protocol, with the disposition of findings 1/2 explicit.
- GPU pilot result containing actual host_cpu_observation, no-hook measurement path, initialized optimizer, actual update receipts and CUDA phase allocation/reservation metrics.
- Fresh-process GPU route-equivalence results at protocol lengths 32 and 128, with matching source pins and full AdamW/model/gradient comparisons.
- External record that no competing GPU work ran; simultaneous CPU load must remain disclosed.
- Do not infer length512/1024 capacity or timing from CPU checks. Preserve OOM/failure and report unchanged batch/precision.
