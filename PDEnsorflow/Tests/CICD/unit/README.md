# Tests/CICD/unit

Tier-1 continuous-integration tests: **fast, CPU-only** checks that run on every
push and pull request. `pytest` collects this folder by default
(`pytest.ini` `testpaths`).

## Current tests

### `test_basewriter.py` &mdash; `gpuSolve.IO.writers.BaseWriter`
Two scenarios (Python `unittest`, discovered by pytest) exercising the
variable-size GPU container that buffers the solution and flushes it in chunks:

* **iteration-based chunking (`every_N`)** &mdash; feeds `n` random tensors and
  checks that solutions stay in memory between flushes, a chunk is dumped only
  on exact multiples of `every_N`, and the final aggregated `.npy` matches the
  input exactly.
* **memory-based chunking (`max_chunk_mb`)** &mdash; with a tiny `1 MB`
  threshold, checks a dump is triggered as soon as the buffer exceeds it and
  that data is preserved across chunks.

The test is autonomous: it creates an isolated temporary directory in `setUp()`
and removes every artefact in `tearDown()`.

### `test_ionic.py` &mdash; `gpuSolve.ionic` cell models
One parametrised contract test over every model (finite, shape-preserving,
deterministic `differentiate()`; a quasi-stable resting state), plus, for the
dimensionless family (`MitchellSchaeffer2v`, `ModifiedMS2v`, `Fenton4v`):

* **[0, 1] rescaling** &mdash; `vmin` maps to 0, `vmax` to 1, and rest is an
  exact fixed point of the potential.
* **retuned range** &mdash; a `vmin`/`vmax` set *after* construction drives the
  rescaling, because the span is derived at every use instead of being cached.
  Guards the defect where a cached span mapped +40 mV to 1.25.
* **per-node range** &mdash; `vmin`/`vmax` may be `(npt, 1)` columns, as pushed
  by `assign_nodal_properties()`, and the span broadcasts.

## Adding tests
Drop a `test_*.py` file here. Keep it **fast and CPU-only** (no GPU assumption,
small problem sizes) so it fits the per-push budget. Heavier or GPU-dependent
regressions belong in `../nightly/`.
