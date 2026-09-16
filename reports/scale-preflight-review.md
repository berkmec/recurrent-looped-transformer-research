# Scale preflight independent review

Verdict: PASS TO START the specified paired three-seed 1,512,706 allocated-parameter parity scale experiment after the GPU owner releases the device. This authorizes a local controlled-task experiment, not scientific closure or large-model validation.

Inspected SPEC.md, integration/run_scale.py, tests/test_integration_scale.py, integration/scale_calibration.json and core/model.py. Existing four-test GREEN evidence was accepted; no unchanged test suite was rerun.

No blocking defect found in protocol freeze, train/validation/test separation, paired initialization or batch construction. All six training runs finish before final tests open. Checkpoints are selected by validation prefix cross entropy with earliest-step ties. No test metric enters training selection. Initial weight hashes must match within each seed. New output/protocol/evidence files fail if already present. Source hashes are frozen and checked after execution. Test predictions/targets/inputs and artifact hashes are saved. Calibration is explicitly training-only.

Resource evidence: width256 calibration reports 1,512,706 allocated and active RLT parameters, versus 1,315,330 active control parameters; median measured step 0.4735 versus 0.02738 seconds, with peak allocated training memory about 102 MB versus 80 MB. These six-step calibration timings are a feasibility estimate only. The core is a one-layer independent tiny implementation with global causal encoder memory and bounded decoder cache; the scale run does not reproduce author-scale training.

Nonblocking reporting requirements: preserve the declared active-parameter and executed-compute mismatch; the no-feedback control uses mathematically equivalent parallel execution while recurrent feedback executes sequentially. Timings are not a compute-matched architecture comparison. Training time includes validation/checkpoint overhead. No evaluation-memory/energy measurement is provided. Width and head changes versus smaller runs prevent attribution solely to parameter count. Correlated prefix positions are not independent samples; perform uncertainty at sequence/seed level. Calibration source_sha256 identifies the model, but the calibration script hash is not recorded in that JSON.

Quality verdict: suitable controlled scale pilot with transparent limits. Specification verdict: scale execution gate passes; full package14 closure still requires completed artifacts, independent metric recomputation and stated external-validity boundaries.
