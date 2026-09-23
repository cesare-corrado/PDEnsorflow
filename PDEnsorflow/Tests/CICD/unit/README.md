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
relative one (`toll = 0`, `toll_rel = 1e-6`), on both the eager per-iteration path
(the default) and the GPU-resident graph loop, and with traced as well as eager
execution. Also covers the **zero system**: a zero right-hand side from a zero
guess must leave X exactly 0 on both paths. That system has a residual of exactly 0, and an unguarded
step length `r.z / p.Ap` would be 0/0 and write NaN, which is what a
pure-diffusion run meets on its first step.

### `test_matrices.py` &mdash; `gpuSolve.matrices.assemble_matrices_dict`
Assembles the mass and stiffness matrices of uniform 1D edge meshes and checks
them entry by entry against their closed tridiagonal forms, plus symmetry,
total mass, zero row sums and positive semi-definiteness. The element entries
are summed on the host, so three more tests pin what that path promises: two
identical assemblies agree **to the bit** (the device sum it replaced did
not), the RCM-renumbered matrix is the plain one with rows and columns
permuted, to the bit, and an element entry missing from the sparsity pattern
raises instead of being summed into a neighbouring entry.

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

### `test_tomek.py` &mdash; the ToR-ORd cell model (`gpuSolve.ionic.tomek.Tomek`)
What is specific to this model beyond the generic contract of `test_ionic.py`.
Tests whose subject does not depend on the integration scheme use forward Euler:
the default tables (with the IKr step matrix on 200001 grid points) take about
2 s to build per model, the forward-Euler ones a fraction of a second. The
default schemes are tested where they are the subject, and through Tomek's
default in `test_ionic.py`, `test_savestate.py` and the parameter-file tests.

* **parameters** &mdash; `celltype` accepts 0 (ENDO), 1 (EPI) and 2 (MCELL) only,
  because a cell-type name is read as ENDO by the reference single-cell tool;
  a constant folded into the lookup tables (e.g. `KNa3`) cannot be set; the
  extracellular `Ko`, `Nao`, `Cao` take one value for the tissue (10 mM `Ko`
  depolarises a resting node; `Nao` set after initialisation rebuilds the table).
* **per-node composition** &mdash; an effective conductance is
  `Coef x cell-type factor x base` (`CoefCaL` acts on `PCa`); an ENDO/EPI/MCELL
  column follows, node by node, the uniform model of each type; one forward
  Euler step of `iF` on an EPI node uses the time constant scaled by
  `delta_epi(V)`, checked against the model equations.
* **physics hooks** &mdash; `GNa = 0` on a node removes its upstroke (both
  integration schemes), as a scar region needs.
* **units and singularities** &mdash; `Cai` is held in mM; the GHK terms, 0/0
  at 0 mV, return their exact limit (L'Hopital) at 0 and within the 1e-6 mV band
  around it, continuous with the formula 1e-3 mV away.
* **the IKr Markov chain** &mdash; in the default mode one step moves the five
  states by `exp(dt Q(V))`, checked against an eigen-decomposition at -80, +20
  and +150 mV; at +860 mV (eigenvalues near -1e17 /ms) the states stay
  nonnegative, keep their total and `O` settles monotonically near 1e-3, where
  forward Euler jumps between the clamps. All four cases fail with the
  forward-Euler chain.
* **forward Euler at high potentials** &mdash; above +300 mV `tm` is below
  1e-16 ms; a gate at its steady state (`m = mL = 1`) must stay there, as it does
  in the reference. Written as `A + B x`, the update cancelled to 0 at +330 mV.
* **front end** &mdash; `imp_region[].im = Tomek` selects the class, and
  `im_param = "celltype=1,GNa=0"` maps per region.

### `test_ionic_plugins.py` &mdash; ionic plugins (`gpuSolve.ionic.plugins`, `IonicModelWithPlugins`)
* **the plugin against the reference step** &mdash; the electroporation current
  and one forward-Euler step of the pore density `n` match a line-by-line NumPy
  transcription of the reference's generated C to 1e-12; `n` starts at its
  steady state for the initial potential.
* **singular points** &mdash; the pore conductance is 0/0 at V = 0 and at
  V = +-935 mV. At and around those points (inside the 1e-6 mV band) it must be
  finite, equal to the analytic limit at 0, and continuous with the formula
  1e-3 mV away. The points +-w0/(nn e/kT) move with the tunable `w0` and `nn`,
  so they are also checked with per-node non-default values, each node exactly
  on its own point (`nn = 0` has none). Removing the band makes these tests fail.
* **precision** &mdash; the plugin works in float64 with a float32 potential.
* **the wrapper** &mdash; `dU` is the model's `dU` minus the plugin current;
  names route to the model (bare) or the plugin (`<class>.<name>`); a plugin
  class is attached once; the per-node `.active` switch removes the current
  node by node while the state is still advanced; `dt` set on the wrapper
  reaches the model and the plugins on initialisation.
* **front end** &mdash; `imp_region[].plugins` (unknown and repeated names are
  refused, a plugin needs a cell model), `plug_param` per region with modifiers,
  the regional switch, malformed `plug_param` lists; a two-region cable run
  attaches the plugin to its region only.
* **checkpoints** &mdash; the checkpoint name is `ModifiedMS2v+<plugin>` and
  restores only into a run with the same plugins; the plugin state survives
  renumbering on a scrambled cable (this fails if the solver looks the state up
  with `getattr` instead of `state_variable()`).

### `test_defib_ashihara_trayanova.py` &mdash; the outward current Ia (`gpuSolve.ionic.plugins.DefibAshiharaTrayanova`)
* **two lower branches** &mdash; the default follows Cheng et al. (1999), Eq. 1,
  to 1e-12 and is continuous (with a continuous slope for `slopeFac = 1`) at
  `VtakeOff` = 100, 160 and 210 mV; the reference form (`set_use_reference_form(True)`)
  matches a NumPy transcription of the reference's generated C to 1e-12, jumps
  221-fold at `VtakeOff`, and equals the default for `VtakeOff = 100` mV.
* **range and precision** &mdash; finite up to 1e5 mV; float64 with a float32
  potential; per-node parameters.
* **interface** &mdash; no state variable (the reference's `Ki` is inert), only
  `VtakeOff` and `slopeFac` accepted; `dU` of the wrapper is the model's minus `Ia`.
* **front end** &mdash; `Defib_AshiharaTrayanova` next to the electroporation
  plugin, `plug_param` per region; a cable run, and a checkpoint named
  `ModifiedMS2v+DefibAshiharaTrayanova` holding only the model's state, which
  restores.

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
`a_crit`, the cell-parameter modifiers (`tau_in*0.3`, `tau_in-10%` and the
other forms, resolved against the cell model default, and the malformed ones
that must raise), the `cg_norm_parab` stopping tests, stimulus defaults derived from
`tend`, and the errors: unknown key, counter that would drop an entry,
non-transmembrane electrode. The **legacy `stimulus[]` keys** must give the same
stimulus as the equivalent `stim[]` keys (box `p0 = x0 - (ctr_def ? xd/2 : 0)`,
`p1 = p0 + xd`), and mixing the two families is refused. The **`flags=` item of
`im_param`** selects the tenTusscherPanfilov cell type of each region (regions
may differ; a combination, an unknown type, or a flag on a model without cell
types is refused), and `GKr` / `GKs` modifiers scale the default of that cell
type, also when a per-node value is pushed before the model is initialised. `meshformat`
and the `lats[]` keys are accepted and reported as not acted upon.

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

### `test_ttp_per_node.py` &mdash; per-node ten Tusscher-Panfilov parameters
* **per-node conductances** &mdash; `GNa`, `GKr`, `GKs` set as `(n, 1)` columns
  reach `differentiate()` as `(n,)` vectors (no `(n, n)` broadcast), and each
  node's current equals that of a uniform model holding its values. `GNa` is a
  float32 tensor (the build of a per-region `GNa*0.0` failed on a Python float);
  a table parameter (`Ko`) refuses non-uniform values.
* **mixed cell types** &mdash; a model with nodes typed EPI, MCELL, ENDO follows
  the three uniform single-type models node by node over 15 ms (GKs, Gto and
  the S-gate time constant switch per node); per-region `flags=` put `celltype`
  first in the parameter map and `GKs*1.5` scales each region's own default.
* **scar end to end** &mdash; `main()` on a 4 mm two-region cable (ENDO half with
  `GNa*1.00,GKr*1.50,GKs*1.50,flags=ENDO`, EPI half with `GNa*0.0`): the build
  succeeds, the healthy half fires, the far millimetre of the scar stays below
  -40 mV, and in the control run (`GNa*1.0`) the same nodes fire.
* **`assign_nodal_properties`** &mdash; the vectorised version gives the same
  columns, values and dtype as the node-by-node loop it replaced, for
  `region`, `nodal` (array and dict) and `uniform` properties.

### `test_savestate.py` &mdash; checkpoints: saving and resuming a run
A run of the paced cable saves its state half way (`tsav`) and every 2 ms
(`chkpt_intv`), then a second run resumes from the half-way state with
renumbering switched on. The restarted output must start with the saved
potential, be recorded on the same steps, and match the uninterrupted run within
`1e-2` mV. It cannot match to round-off: the CG warm-start history `U^{n-1}` is
not in the file, so the first step after the restart starts CG from a different
guess.

Also covers: every tf.Variable that `differentiate()` changes is declared by
`state_variable_names()`, for all six cell models; a cell model compiled before
`finalize_for_run()` renumbers its state advances the renumbered variables, and
matches a run compiled afterwards; a ten Tusscher-Panfilov state
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
