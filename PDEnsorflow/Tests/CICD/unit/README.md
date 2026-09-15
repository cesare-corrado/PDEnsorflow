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

### `test_conjgrad.py` &mdash; `gpuSolve.linearsolvers.ConjGrad`
Assembles `A = M + K` on the coarse square mesh, prescribes a solution
(all ones, and a fixed random vector), and checks that the Jacobi-preconditioned
CG recovers it, once through the absolute tolerance and once through the
relative one (`toll = 0`, `toll_rel = 1e-6`). Also covers the **zero system**: a
zero right-hand side from a zero guess must leave X exactly 0 on both the eager
and the graph path. That system has a residual of exactly 0, and an unguarded
step length `r.z / p.Ap` would be 0/0 and write NaN, which is what a
pure-diffusion run meets on its first step.

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

Also covers the **vertex-file electrode**: a short run whose `elec.vtx_file`
names five nodes must depolarise exactly those and leave the far end at rest
(over 2 ms the diffusion length is ~700 um, so the far end cannot be reached),
and an index outside the mesh must be refused rather than wrapping round.

A **pure-diffusion run** (no cell model) must stay finite: it starts from
`U = 0` with nothing driving it, so its output must stay exactly 0 rather than
turning into NaN on the first step.

### `test_savestate.py` &mdash; checkpoints: saving and resuming a run
A run of the paced cable saves its state half way (`tsav`) and every 2 ms
(`chkpt_intv`), then a second run resumes from the half-way state with
renumbering switched on. The restarted output must start with the saved
potential, be recorded on the same steps, and match the uninterrupted run within
`1e-2` mV. It cannot match to round-off: the CG warm-start history `U^{n-1}` is
not in the file, so the first step after the restart starts CG from a different
guess.

Also covers: every tf.Variable that `differentiate()` changes is declared by
`state_variable_names()`, for all five cell models; a ten Tusscher-Panfilov state
on a scrambled mesh survives the renumbering for every variable; a pure-diffusion
run saves and resumes; the file round-trips and a damaged one is refused; a
checkpoint from another cell model or mesh, one saved after `tend`, or a missing
file stops the run; and the `savestate` defaults and ranges of the parameters.

### `test_mesh_roundtrip.py` &mdash; the external mesh format
Writes a small strip through `Triangulation.exportCarpFormat` and reads it back.
Pins the line-element specifier: the writer emits `Ln`, which is what the
`.elem` format defines and what other readers of it expect, while the reader
still accepts the `Cx` this package used to write, so meshes already on disk
keep loading.

### `test_vtxreader.py` &mdash; `.vtx` vertex specification files
A `.vtx` file names a set of nodes outright: a count, an optional
`intra` / `extra` keyword, then one 0-based index per line. Covers the two
tolerances that make such a file portable (the keyword is neither counted nor
required; reading stops at the declared count so trailing content is ignored)
and the three refusals: a file shorter than it declares, one carrying a second
per-node column, and one with no count.

## Adding tests
Drop a `test_*.py` file here. Keep it **fast and CPU-only** (no GPU assumption,
small problem sizes) so it fits the per-push budget. Heavier or GPU-dependent
regressions belong in `../nightly/`.
