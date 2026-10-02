# MLX CI compatibility repair

Date: 2026-10-02. Engineering maintenance, not a new research mechanism.

## Diagnosis and intended repair, recorded before implementation

GitHub run 37044963944 on db157a4 fails 29 training tests with MLX 0.32.3;
581 tests and 13 subtests pass. Local verification used MLX 0.32.0. A fresh
temporary environment with the CI versions reproduces the same gather-index
gradient exception in `test_emergent_language.py`.

The affected message encoders select hard tokens by indexing an identity
matrix with model-derived argmax indices. Their straight-through estimator
intends the hard choice to have zero derivative and the soft probabilities to
carry the gradient. Make that discrete boundary explicit with `stop_gradient`
on indices, preserving the existing forward arithmetic and soft derivative.
Share the operation across the affected encoders so this contract has one
implementation. Do not change objectives, thresholds, experiment gates or
skip failing tests.

Verification requires a regression test of hard forward values and the
analytic softmax gradient, the affected training tests on 0.32.3, the full
suite on both local 0.32.0 and CI 0.32.3, and a passing GitHub run. Record the
tested CI dependency versions in a constraints file and installation output
so a fresh CI install is reproducible. Published experiment results remain
records of their original code and environment; no scientific rerun is claimed.


## Implemented repair and local verification

The common operation is `homesocial.discrete.straight_through_one_hot`.
Six legacy encoder modules use it. Dialogue repair indices and frozen-partner
choices also stop gradients explicitly before indexing. Forward arithmetic
retains the original order. Four regression cases verify exact legacy forward
values, tie behavior, batched shapes and the analytic softmax/identity
surrogate gradients; gradients remain nonzero so sender learning is preserved.

The fresh 0.32.3 reproduction failed before this change. Afterwards all 59
focused training/regression tests pass. The full suite passes **614 tests and
13 subtests** on both **MLX 0.32.3 (71.36 seconds)** and the research
environment's **0.32.0 (72.66 seconds)**. Public result checkers for Probes63,
64, 65, 75, 77, 78 and 79 still pass. No tests are skipped or weakened.

`constraints-ci.txt` records all installed runtime/test dependencies; the
workflow uses it for installation and cache invalidation, runs `pip check`,
and prints installed versions. Project metadata retains its supported version
ranges. README setup uses the same constraints. Python is still the maintained
3.13 series; this file does not lock the runner OS or build backend, and is
not a promise of bitwise reproducibility of all training across hardware.

This repairs compatibility, without reopening any closed research line or
changing the interpretation of published experiment results. GitHub Actions
is the authoritative record of remote verification for the pushed commit.
