# P7 GPU-loss Phase-A recovery receipt

The executable now maps a latched wgpu device loss to the distinct process exit
code `75`, after the normal event-loop exit hook performs the bounded recovery
writer flush and decoder-retirement drain. `scripts/supervise-collide-o-scope.ps1`
may relaunch that code once and only once. The next process exposes the existing
explicit recovery-journal UI; it does not restore a patch, rebind a source or
monitor, reopen an audience endpoint, or resume Program automatically.

`src/gpu_recovery.rs` freezes the full epoch law before any Phase-B work:
Healthy → ClosingAudience → RetiringGpuEpoch → Rebuilding → AwaitingOperator,
or the retained Phase-A SupervisedRestartRequired branch. Epoch-qualified
completions from the retired device are rejected. Required claims must validate
before AwaitingOperator, operator resume is explicit, rebuild attempts are
capped at one, and a second loss while recovering terminates Failed rather than
recursing. Deterministic tests exercise all eight named injection seams.

The injection tests are synchronization/state-machine proof, not real driver
resets. A launcher deadline and actual relaunch-to-recovery-surface measurement
have not been run on a packaged binary, and in-process resource rebuilding is
not enabled. The audit's stop rule is therefore applied: retain the simpler
supervised restart and do not claim transparent GPU continuity.
