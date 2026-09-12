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

### `test_optionreader.py` &mdash; the `.par` lexer and the command line
Pure text handling, no mesh and no TensorFlow, so it runs in hundredths of a
second: comments inside and outside quotes, backslash continuations, quoted and
bare values, splitting on the first `=`, the indexed and aggregate tag-list
forms, the `-key` / `--key` / `--key=value` spellings, and `+Save` round-tripping
to an identical store. The central case is **resolution order**: a flag before
`+F` loses to the file and one after it wins, which is the format's rule and the
opposite of what "the command line overrides the file" would give.

### `test_parametermapper.py` &mdash; openCARP keys to gpuSolve settings
The three things that fail silently if they are wrong: the **defaults** (a key
that is absent means what that format says it means, except cell parameters,
which keep the gpuSolve class default); the **unit conversion**
`sigma = 1e5 g g_mult / (beta volFrac)` from S/m and micrometres to um^2/ms; and
the **conductivity rule**, since `bidm_eqv_mono` defaults to 1 and makes the
monodomain conductivity the half harmonic mean of the two domains rather than
`g_il`. Also covers which region governs which tag, cell-model selection by
`a_crit`, the `cg_norm_parab` stopping tests, stimulus defaults derived from
`tend`, and the errors: unknown key, counter that would drop an entry,
non-transmembrane electrode.

### `test_carp_compatibility_run.py` &mdash; the front end end to end
One short run of `main()` on a 5 mm cable written as `.pts` / `.elem` / `.lon`
with `Ln` line elements, driving the whole chain from the parameter file to the
IGB output. Checks the exit status, that `+Save` wrote the resolved set, that the
IGB header matches the frames actually recorded, that the potential stays finite
and inside the model's band, and that the activation front travels in one
direction. The quantitative conduction-velocity check lives with the example in
`Tests/FEM/mMS_carp_compatibility`, where the mesh resolves the front.

## Adding tests
Drop a `test_*.py` file here. Keep it **fast and CPU-only** (no GPU assumption,
small problem sizes) so it fits the per-push budget. Heavier or GPU-dependent
regressions belong in `../nightly/`.
