# Resolved R6 context for implementation

Use task-3-brief.md first, then this context. Immutable Core source is in this
worktree; no changes there. Existing CUDA Python verified read-only:
C:/Users/pc/Documents/ChatGPT/DaxDream/.venv/Scripts/python.exe,
torch2.12.0+cu130, RTX2060SUPER8GB. No installs/downloads. GPU reservation must
be coordinated with parent before any pilot or main. Main R5 will be GPU serial.

Recommended declared arms (do not hide route changes): full, detach-only TBPTT8,
whole-forward checkpoint full, segment-backward TBPTT8, segment-backward plus
encoder checkpoint TBPTT8. The last arm can be omitted only with an explicit
scope ruling and reason; test enough to determine if it is meaningful.

Important route: immutable Core forward encodes all tokens first; mk/mv global
paths remain attached across original TBPTT boundaries, while s/dk/dv detach.
A possible exact segment implementation is: compute global e/mk/mv once; make
detached leaf views for decoder inputs; backward each eight-token loss scaled by
the SAME total token count; accumulate the three leaf gradients; finally
backpropagate those gradients through the original encoder/projection graph in
one multi-output backward. Detach only decoder state s/dk/dv at each boundary.
This preserves global parameter routes and frees segment decoder graphs. This
is a proposed route, not a presupposed correctness result; independently verify
against detach-only model gradients and actual AdamW states before measurement.

Default dev/main design target: width32,heads2,batch8,float32,parity dense loss;
length ladder32/128/512/1024, chunk8; same settings across arms. Two warmup
optimizer updates and three measured updates per fresh subprocess/condition.
If development shows this ladder too costly/infeasible, propose a concrete
resource adjustment BEFORE main freeze, never silently shrink OOM runs. OOM
should remain recorded. Saved-tensor hooks diagnostics may distort latency;
keep primary CUDA memory/timing runs separate from explicitly diagnostic hook
runs and deduplicate backing storage. Record process RAM as a separate metric.

Main and any equivalent route validation must use fresh processes for allocator
isolation. No shell dependency on activation scripts; explicit Python path.
Preserve all failed pilots, source/protocol pins, stderr and process IDs.
Parent will review/coordinate before main; implementation owner may run bounded
CPU/GPU development checks only after the GPU notification.
