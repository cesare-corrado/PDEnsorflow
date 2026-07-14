# Tests/CICD/nightly

Tier-2 continuous-integration tests: **heavier, reduced-config** solver
regressions with numeric sanity checks (e.g. conduction velocity vs the analytic
front speed, finite solution fields, potential inside the physical band). These
run on a **scheduled** basis on a **self-hosted GPU runner** (see
`.github/workflows/nightly.yml`), not on every push.

These tests are added incrementally, typically by taking the profiling scripts in
`Tests/DEVTESTS` and the examples in `Tests/FD` / `Tests/FEM`, shrinking the
problem size / number of steps, and adding assertions.

`pytest` does **not** collect this folder by default (`pytest.ini` points
`testpaths` at `unit/` only); the nightly workflow selects it explicitly. On the
GPU box the whole suite (Tier-1 unit + Tier-2 nightly) is run together, so the
device-gated `csr_axpby` native-path cases that skip on the CPU build actually
execute:

```bash
# on the self-hosted GPU runner (conda env with the CUDA activate.d shim)
conda run -n <gpu-env> python -m pytest PDEnsorflow/Tests/CICD/unit PDEnsorflow/Tests/CICD/nightly -v
```

## Contents

- `test_mms_2d_regression.py` &mdash; 2-D modified Mitchell-Schaeffer monodomain
  regression on the real `Tests/data/triangulated_square.pkl` sheet (63001 nodes,
  RCM renumbering). A planar front is launched at one edge; two tests share a
  single module-scoped run and check (a) the front is unidirectional (per-x-column
  activation times increase monotonically, potential stays in `[-80, 20]` mV) and
  (b) the measured conduction velocity matches the analytic Nagumo speed
  `CV = 0.5 (1 - 2 u_crit) sqrt(2 sigma / tau_in)` within 10% (~4.2% at the tuned
  parameters). Marked `nightly` + `gpu`.
- `_gpu_check.py` &mdash; GPU sanity gate for the workflow (imports gpuSolve, then
  fails if no physical GPU is visible). Not a test; the leading underscore keeps
  pytest from collecting it. Run it as a script, never via `python -c`.
