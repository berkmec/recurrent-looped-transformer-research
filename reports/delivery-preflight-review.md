# Independent delivery preflight

Reviewed the first version of `audit/delivery_integrity.py`, its four tests, `reports/user-requirement-map.md`, and the finite-state-information addition to the mathematical note. No actual campaign manifest was built or verified. Temporary reproductions used disposable directories outside the campaign and were cleaned up.

The inventory covers ignored `.pt`, `.npz`, and `.sqlite` payloads; the tests cover same-size byte replacement, missing/extra payloads, cache exclusions, refusal to overwrite, and incomplete status. Exclusive writes preserve an existing manifest. Digest binding is explicitly internal integrity, not protection against privileged replacement of both bytes and manifest.

Two implementation hardening findings were sent to the owner before sealing:

1. **Reproduced: a previously hashed file can change before build completion without rejection.** The per-file before/after stat covers its own hash interval; the final inventory pass checks filenames only. In a temporary fixture, modifying `a.npz` while hashing later `z.pt` caused `build_manifest` to report success with the stale digest. Immediate `verify_manifest` correctly rejected it. Retain and compare the complete stat snapshot after all hashes, and require successful verification after construction. The owner's external writer-quiescence requirement avoids this specific live-writer scenario but should not be replaced with a claim that construction detects every concurrent write.
2. **Static containment weakness:** closure and sidecar paths are opened without their own resolved containment check, while inventory containment applies only to included files. On runtimes that omit directory symlinks during recursive traversal, an `audit` directory link can redirect sidecar output. A native symlink reproduction was unavailable because Windows denied symlink creation with error 1314; no actual campaign path escape was observed. Explicitly reject directory links/junctions and check resolved closure/sidecar containment before any read/write.

These concern the delivery utility; neither is a finding that existing scientific payloads are invalid. A status string alone cannot establish that a missing expected scientific file once existed. Required-artifact completeness remains the responsibility of the independent experiment audits and the owner's final campaign closure. The manifest preserves the inventory present at sealing and subsequently detects deviations from it.

The user requirement map and finite-state-information paragraph pass scope review. The paragraph correctly explains that parity and five-state tasks admit a bounded sufficient state and do not establish unbounded information capacity. The map distinguishes implementation from general superiority, reports failed vector retrieval, limits generated-state continuation to the bounded setup, and treats trajectory distillation gains as relative to its paired answer-only arm. Final companion result claims still require its outstanding completion audit.

No experimental source or producer-owned file was changed by this review. This file records the first reviewed implementation; owner fixes and passing regressions may supersede the two hardening findings.

## Re-review of owner fixes

**PASS for the declared quiescent-writer sealing workflow.** Reviewed delivery utility SHA256 `ac1b946a1b8a028a53eacf1be0fa3e4a833a5890b5ea5b09658a2b4ab39cda72`. The implementation retains each hashed file's size/mtime signature, checks the full inventory and all signatures again after hashing, and requires a complete verification before construction returns success. The previously reproduced late-file mutation is now rejected before a manifest is created. Manifest, digest, and closure access have resolved root-containment checks. Inventory explicitly refuses symlinks and Windows reparse points, including junction directories.

Independently reran the bounded test file: **5 passed in 0.20 seconds**, including the late-file race regression. No actual campaign seal was created. Both original hardening findings are closed for the intended workflow. Writer quiescence remains an explicit precondition; the utility is not an atomic filesystem snapshot or an adversarial concurrency/security mechanism. Required scientific artifact completeness still comes from campaign closure and independent experiment audits.

All review writes completed with this addendum; no background process or open writer was left running.
