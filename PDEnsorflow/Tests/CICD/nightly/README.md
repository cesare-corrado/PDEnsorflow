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
  `CV = 0.5 (1 - 2 u_crit) sqrt(2 sigma / tau_in)` within 10% (~2.0% at the tuned
  parameters with the default Crank-Nicolson step, ~4.2% with implicit Euler).
  Marked `nightly` + `gpu`.
- `test_carp_compatibility_2d_regression.py` &mdash; the same physics on the same
  sheet, driven through a **`.par` parameter file** and
  `gpuSolve.carp_compatibility.main` instead of the Python API. The fixture
  writes the sheet as `.pts` / `.elem` / `.lon` with coordinates in
  **micrometres** (the unit that format specifies) into a temporary directory,
  and the parameter file expresses everything in that format's units: `g_il` in
  S/m, `cellSurfVolRatio` in um^-1, `dt` in microseconds. The two tests check the
  front is unidirectional and that the conduction velocity matches the same
  analytic Nagumo speed within 10%.

  It earns its GPU minutes because it covers layers a small cable cannot reach:
  the **vectorised assembly path**, which builds the diffusion tensor from the
  region maps *without* calling the material function and therefore has to apply
  `beta` itself; RCM renumbering of a real mesh; and the IGB writer's chunked
  flush. Two defects of exactly that kind were found by running the real example
  rather than the unit suite. Marked `nightly` + `gpu`.
- `test_tomek_regression.py` &mdash; one paced beat of the Tomek (ToR-ORd) model
  for ENDO, EPI and MCELL (three nodes of one model), with the default schemes
  (Rush-Larsen gates, matrix-exponential IKr chain) and with forward Euler (the
  reference's scheme), at `dt = 0.01` ms. Peak Vm, APD90 and the `Cai` peak are checked
  against the reference single-cell tool (`bench --imp Tomek`, same protocol;
  the command is in the module docstring) within 1 mV, 1 ms and 2%. The stepping
  loop is compiled with XLA, which brings the beat from about 20 minutes to
  about 15 seconds. Marked `nightly` + `gpu`.
- `test_electroporation_regression.py` &mdash; the electroporation plugin
  (`ElectroporationDeBruinKrassowska98`) attached through
  `IonicModelWithPlugins`, against the reference single-cell tool (`bench`, the
  commands are in the module docstring) at `dt = 0.01` ms: Tomek at rest for
  50 ms (the plugin's leak makes the cell fire at about 48 ms), and a passive
  membrane under a 1000 uA/uF shock that drives V to +470.9 mV. V and the pore
  density are checked at fixed times within 0.05 mV and 1e-6. The shock uses a
  passive parent so that it tests the plugin alone: above +222 mV (at
  `dt = 0.01` ms) forward Euler on Tomek's IKr Markov chain is unstable in both
  codes, so that part of a Tomek trajectory is a shared numerical artifact.
  Marked `nightly` + `gpu`.
- `_gpu_check.py` &mdash; GPU sanity gate for the workflow (imports gpuSolve, then
  fails if no physical GPU is visible). Not a test; the leading underscore keeps
  pytest from collecting it. Run it as a script, never via `python -c`.
